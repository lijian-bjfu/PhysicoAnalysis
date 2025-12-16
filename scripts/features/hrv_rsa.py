import pandas as pd
import numpy as np
from typing import Optional, Tuple

import sys
from pathlib import Path
# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---
from settings import PARAMS

from scipy.signal import find_peaks

"""
RSA（窦性呼吸性心律不齐）特征（单段 RR）

设计原则（与 hrv_time/hrv_freq 对齐）：
- 仅对“单段 RR + 可选呼吸”计算 RSA，不做切窗/清洗/批处理。
- 公共接口与其它特征模块一致：features_segment(rr_df, resp_df=None) -> DataFrame[1xK]
- 以“峰-谷（peak-to-valley）”为主方法；若无呼吸，则返回 NaN。
- 支持含 NaN 的断续呼吸：按 NaN 切分，逐段计算后汇总（RSA 均值、rate 中位数）。
- 所有口径与阈值从 settings.PARAMS 读取（单一真相源）。

输出字段（最小且可用于论文的方法段）：
- rsa_ms            : RSA（毫秒），按每一呼吸周期的 (max RR - min RR) 取平均；若无足够呼吸周期则 NaN
- resp_rate_bpm     : 本段估计的平均呼吸频率（次/分）；无呼吸或不足则 NaN
- n_breaths_used    : 计入 RSA 计算的呼吸周期数量
- resp_amp         : 呼吸振幅（每个呼吸周期的峰-谷差的聚合，单位为原始呼吸信号幅值；无有效周期则 NaN)
- resp_log_amp     : 呼吸振幅的自然对数 ln(resp_amp)；若 ≤0 或 NaN 则返回 NaN
- rsa_method        : 'peak_to_valley' 或 'unavailable'

说明：
- 我们将 RR 间期（不等间隔）转为心搏时间戳，用呼吸峰对本段分割为一个个呼吸周期；
  每个周期内取 RR 的最大值与最小值之差（ms），作为该周期的 RSA，最后对所有有效周期取平均。
- 呼吸峰检测基于 scipy.signal.find_peaks；周期时长门槛（对应 0.1–0.5 Hz ≈ 6–30 次/分）防止非生理峰干扰。
- 若需与 HF 对齐，可在统计分析阶段把 RSA 与 HF_log 一同报告；本函数只负责计算 RSA。
"""

# ---------- 内部工具 ----------

def _rr_to_beat_times(rr_ms: np.ndarray) -> np.ndarray:
    """将 RR(ms) 转为心搏发生时间（秒），首个时间从 0 开始。"""
    if rr_ms.size == 0:
        return np.array([])
    t = np.cumsum(rr_ms) / 1000.0
    t -= t[0]
    return t


def _estimate_resp_rate_from_peaks(t_s: np.ndarray, peaks_idx: np.ndarray) -> float:
    if peaks_idx.size < 2:
        return np.nan
    periods = np.diff(t_s[peaks_idx])
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if periods.size == 0:
        return np.nan
    # 使用中位数更稳健
    return 60.0 / float(np.median(periods))


def _compute_resp_amp(t_resp: np.ndarray, x_resp: np.ndarray, peaks_idx: np.ndarray, agg: str = 'median') -> float:
    """
    计算呼吸振幅（单位=原始呼吸信号幅值）。定义为：
    在相邻两个呼吸峰之间，取该区间内的信号最大值减最小值（peak-to-trough）作为该周期振幅，
    对所有有效周期做聚合（median 或 mean）。若无有效周期返回 NaN。
    """
    if peaks_idx.size < 2:
        return np.nan
    amps = []
    for i in range(peaks_idx.size - 1):
        p0, p1 = peaks_idx[i], peaks_idx[i + 1]
        # 取两个峰之间的原始信号段
        seg = x_resp[p0:p1 + 1]
        if seg.size == 0 or not np.isfinite(seg).any():
            continue
        a = float(np.nanmax(seg) - np.nanmin(seg))
        amps.append(a)
    if not amps:
        return np.nan
    amps = np.array(amps, dtype=float)
    if agg == 'mean':
        return float(np.nanmean(amps))
    else:
        return float(np.nanmedian(amps))


# ---------- 公共 API ----------

def _split_valid_spans(
    t: np.ndarray,
    x: np.ndarray,
    *,
    split_on_time_gap: bool = False,
    max_gap_s: float = np.inf,
) -> list[tuple[int, int]]:
    """Split into contiguous spans where x is finite.

    Default behavior (split_on_time_gap=False):
    - ONLY uses NaN/inf in x (or t) as span breaks.

    Optional safeguard (split_on_time_gap=True):
    - ALSO breaks spans when the gap between consecutive timestamps is larger than max_gap_s.

    Returns a list of (start_idx, end_idx) inclusive.
    """
    if t.size == 0:
        return []

    finite = np.isfinite(t) & np.isfinite(x)
    spans: list[tuple[int, int]] = []
    n = t.size
    i = 0

    while i < n:
        if not finite[i]:
            i += 1
            continue
        j = i
        while j + 1 < n:
            if not finite[j + 1]:
                break
            if split_on_time_gap and (t[j + 1] - t[j]) > max_gap_s:
                break
            j += 1
        spans.append((i, j))
        i = j + 1

    return spans


def features_segment(rr_df: pd.DataFrame, resp_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    计算单段 RR 的 RSA（peak-to-valley）。

    参数
    rr_df : DataFrame，列 ['t_s','rr_ms']；已去伪迹的 NN 间期（毫秒）。
    resp_df : （可选）DataFrame，列 ['t_s','resp']；与 RR 段对齐的呼吸信号。

    返回
    单行 DataFrame：['rsa_ms','resp_rate_bpm','n_breaths_used','rsa_method']。
    若无呼吸或有效呼吸周期不足，返回 NaN，并标注 rsa_method='unavailable'。
    """
    # —— 读取参数（全部来自 PARAMS）——
    # 呼吸峰检测：最小峰距（秒）≈ 上限呼吸率（30 次/分 -> 2 秒/次）的 80% 安全边际
    resp_min_bpm = float(PARAMS.get('resp_min_bpm', 6.0))     # 对应 0.1 Hz
    resp_max_bpm = float(PARAMS.get('resp_max_bpm', 30.0))    # 对应 0.5 Hz
    resp_peak_prominence = float(PARAMS.get('resp_peak_prominence', 0.0))  # find_peaks 的 prominence，下游可按需要调
    min_rr_per_breath = int(PARAMS.get('rsa_min_rr_per_breath', 2))       # 每个呼吸周期内的最少 RR 个数
    rsa_agg = str(PARAMS.get('rsa_agg', 'mean'))                           # 'mean' 或 'median'

    # 基本容错：呼吸列允许 'resp' 或 'value'
    resp_col = None
    if resp_df is not None and not resp_df.empty:
        if 'resp' in resp_df:
            resp_col = 'resp'
        elif 'value' in resp_df:
            resp_col = 'value'

    if resp_df is None or resp_df.empty or ('t_s' not in resp_df) or (resp_col is None):
        return pd.DataFrame([{
            'rsa_ms': np.nan,
            'rsa_log_ms': np.nan,
            'resp_amp': np.nan,
            'resp_log_amp': np.nan,
            'resp_rate_bpm': np.nan,
            'n_breaths_used': 0,
            'rsa_method': 'unavailable'
        }])

    if rr_df is None or rr_df.empty or ('rr_ms' not in rr_df):
        return pd.DataFrame([{
            'rsa_ms': np.nan,
            'rsa_log_ms': np.nan,
            'resp_amp': np.nan,
            'resp_log_amp': np.nan,
            'resp_rate_bpm': np.nan,
            'n_breaths_used': 0,
            'rsa_method': 'unavailable'
        }])

    # —— 可选：对本段呼吸时间做 rebase 到 0，便于与 RR 的 t_beats(从0开始)对齐 ——
    if bool(PARAMS.get('rsa_rebase_resp', True)):
        resp_df = resp_df.copy()
        resp_df['t_s'] = resp_df['t_s'] - float(resp_df['t_s'].iloc[0])

    # —— 呼吸峰检测 ——
    t_resp = resp_df['t_s'].to_numpy(dtype=float)
    x_resp = resp_df[resp_col].to_numpy(dtype=float)
    if t_resp.size < 3:
        return pd.DataFrame([{
            'rsa_ms': np.nan,
            'rsa_log_ms': np.nan,
            'resp_amp': np.nan,
            'resp_log_amp': np.nan,
            'resp_rate_bpm': np.nan,
            'n_breaths_used': 0,
            'rsa_method': 'unavailable'
        }])

    # 估计采样间隔与峰距约束
    dt = np.median(np.diff(t_resp)) if t_resp.size > 1 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return pd.DataFrame([{
            'rsa_ms': np.nan,
            'rsa_log_ms': np.nan,
            'resp_amp': np.nan,
            'resp_log_amp': np.nan,
            'resp_rate_bpm': np.nan,
            'n_breaths_used': 0,
            'rsa_method': 'unavailable'
        }])

    # —— 处理清理后的“断续/NaN”呼吸：按 NaN 切分为多个连续片段，逐段计算后汇总 ——
    # 设计选择：默认只看 NaN（clean_resp 对不连续/无效段负责插 NaN）。
    split_on_time_gap = bool(PARAMS.get('rsa_split_on_time_gap', False))
    gap_factor = float(PARAMS.get('resp_gap_factor', 3.0))
    max_gap_s = float(PARAMS.get('resp_max_gap_s', gap_factor * dt))

    spans = _split_valid_spans(
        t_resp,
        x_resp,
        split_on_time_gap=split_on_time_gap,
        max_gap_s=max_gap_s,
    )

    if not spans:
        return pd.DataFrame([{
            'rsa_ms': np.nan,
            'rsa_log_ms': np.nan,
            'resp_amp': np.nan,
            'resp_log_amp': np.nan,
            'resp_rate_bpm': np.nan,
            'n_breaths_used': 0,
            'rsa_method': 'unavailable'
        }])

    # 统一的“最低可算”门槛：至少需要 2 个呼吸周期
    rsa_min_breaths = int(PARAMS.get('rsa_min_breaths', 2))
    min_iqr = float(PARAMS.get('rsa_min_resp_iqr', 1e-6))

    # —— RR → 心搏时间 ——
    rr = rr_df['rr_ms'].to_numpy(dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size == 0:
        return pd.DataFrame([{
            'rsa_ms': np.nan,
            'rsa_log_ms': np.nan,
            'resp_amp': np.nan,
            'resp_log_amp': np.nan,
            'resp_rate_bpm': np.nan,
            'n_breaths_used': 0,
            'rsa_method': 'unavailable'
        }])

    # 优先使用 rr_df['t_s']（若可用），以更稳健地与 resp 的时间轴对齐；否则回退到累加 RR。
    if 't_s' in rr_df and np.isfinite(rr_df['t_s'].to_numpy(dtype=float)).any():
        t_beats = rr_df['t_s'].to_numpy(dtype=float)
        # 与 resp 的 rebase(从窗口起点)对齐：把首个心搏时间归零
        t0b = float(t_beats[0]) if np.isfinite(t_beats[0]) else float(np.nanmin(t_beats))
        if np.isfinite(t0b):
            t_beats = t_beats - t0b
        # 只保留与 rr 同步的有限时间戳
        good = np.isfinite(t_beats)
        t_beats = t_beats[good]
        rr = rr[good]
    else:
        t_beats = _rr_to_beat_times(rr)

    # 收集每个 span 的结果，用于窗口级汇总
    span_rsa_values: list[float] = []
    span_rate_values: list[float] = []
    span_amp_values: list[float] = []
    breaths_used_total = 0

    # —— 逐 span 找峰并计算 RSA（每段单独计算，最后汇总）——
    for (i0, i1) in spans:
        # 片段长度检查
        if i1 <= i0 or (i1 - i0 + 1) < 5:
            continue

        t_span = t_resp[i0:i1 + 1].astype(float)
        x_span = x_resp[i0:i1 + 1].astype(float)

        # 低振幅/锁死保护：IQR 过小基本不可能可靠找峰
        q75_s, q25_s = np.nanpercentile(x_span, [75, 25])
        iqr_s = float(q75_s - q25_s)
        if (not np.isfinite(iqr_s)) or (iqr_s <= min_iqr):
            continue

        # —— 轻度预处理（可选）：仅用于稳健找峰，不改变 RSA 定义 ——
        x_filt = x_span.copy()
        smooth_sec = float(PARAMS.get('rsa_resp_smooth_sec', 0.0))
        if smooth_sec > 0:
            win = max(1, int(round(smooth_sec / dt)))
            if win % 2 == 0:
                win += 1
            if win > 1:
                ker = np.ones(win, dtype=float) / float(win)
                x_filt = np.convolve(x_filt, ker, mode='same')

        # 动态显著性：若未指定（=0），用 IQR 的一小部分抑制毛刺
        if resp_peak_prominence and resp_peak_prominence > 0:
            prom = resp_peak_prominence
        else:
            prom = 0.1 * iqr_s if np.isfinite(iqr_s) and iqr_s > 0 else 0.0

        # 基于生理范围设置最小峰距（样本点）
        min_dist_sec = 60.0 / resp_max_bpm
        distance_pts = max(1, int(np.floor(min_dist_sec / dt * 0.8)))

        peaks_pos, _ = find_peaks(x_filt, distance=distance_pts, prominence=prom)
        peaks_neg, _ = find_peaks(-x_filt, distance=distance_pts, prominence=prom)

        def _valid_with_tail_span(peaks_idx: np.ndarray) -> np.ndarray:
            """Select a physiologically-valid contiguous peak sequence.

            Keep the longest contiguous subsequence whose consecutive peak-to-peak BPM
            stays within [resp_min_bpm, resp_max_bpm]. This avoids the previous
            “every-other-peak” behavior that can double the median period and halve
            resp_rate_bpm.
            """
            if peaks_idx.size < 2:
                return np.array([], dtype=int)

            t_peaks = t_span[peaks_idx]
            periods = np.diff(t_peaks)

            good = np.isfinite(periods) & (periods > 0)
            bpm = np.full(periods.shape, np.nan, dtype=float)
            bpm[good] = 60.0 / periods[good]
            valid_edge = good & (bpm >= resp_min_bpm) & (bpm <= resp_max_bpm)

            if not np.any(valid_edge):
                return np.array([], dtype=int)

            # Find the longest run of True in valid_edge.
            # valid_edge[i] refers to edge between peaks_idx[i] and peaks_idx[i+1].
            best_start = 0
            best_end = -1
            cur_start = None

            for i, ok in enumerate(valid_edge):
                if ok:
                    if cur_start is None:
                        cur_start = i
                else:
                    if cur_start is not None:
                        cur_end = i - 1
                        if (best_end - best_start) < (cur_end - cur_start):
                            best_start, best_end = cur_start, cur_end
                        cur_start = None

            if cur_start is not None:
                cur_end = len(valid_edge) - 1
                if (best_end - best_start) < (cur_end - cur_start):
                    best_start, best_end = cur_start, cur_end

            # Convert edge-run [best_start..best_end] to peak-run [best_start..best_end+1]
            return peaks_idx[best_start:best_end + 2].astype(int)

        vpos = _valid_with_tail_span(peaks_pos)
        vneg = _valid_with_tail_span(peaks_neg)
        cycles_pos = max(0, vpos.size - 1)
        cycles_neg = max(0, vneg.size - 1)
        valid_peaks = vpos if cycles_pos >= cycles_neg else vneg

        if valid_peaks.size < 2:
            continue

        # —— 在每个呼吸周期内计算 (max RR - min RR) ——
        rsa_values = []
        breath_count = 0
        for k in range(valid_peaks.size - 1):
            t0 = t_span[valid_peaks[k]]
            t1 = t_span[valid_peaks[k + 1]]
            sel = (t_beats >= t0) & (t_beats < t1)
            rr_in = rr[sel]
            if rr_in.size >= min_rr_per_breath:
                rsa_values.append(float(np.max(rr_in) - np.min(rr_in)))
                breath_count += 1

        if breath_count < rsa_min_breaths:
            continue

        rsa_values = np.array(rsa_values, dtype=float)
        if rsa_agg == 'median':
            rsa_span = float(np.median(rsa_values))
        else:
            rsa_span = float(np.mean(rsa_values))

        rate_span = _estimate_resp_rate_from_peaks(t_span, valid_peaks)
        amp_span = _compute_resp_amp(t_span, x_span, valid_peaks, agg=rsa_agg)

        if np.isfinite(rsa_span):
            span_rsa_values.append(rsa_span)
        if np.isfinite(rate_span):
            span_rate_values.append(rate_span)
        if np.isfinite(amp_span):
            span_amp_values.append(amp_span)

        breaths_used_total += int(breath_count)

    # —— 汇总（窗口级）：RSA 取均值，呼吸率取中位数 ——
    if len(span_rsa_values) == 0:
        # 没有任何可用片段
        return pd.DataFrame([{
            'rsa_ms': np.nan,
            'rsa_log_ms': np.nan,
            'resp_amp': np.nan,
            'resp_log_amp': np.nan,
            'resp_rate_bpm': float(np.nanmedian(span_rate_values)) if len(span_rate_values) else np.nan,
            'n_breaths_used': int(breaths_used_total),
            'rsa_method': 'unavailable'
        }])

    rsa_ms = float(np.nanmean(np.array(span_rsa_values, dtype=float)))
    rsa_log_ms = np.log(rsa_ms) if (np.isfinite(rsa_ms) and rsa_ms > 0) else np.nan

    resp_rate_bpm = float(np.nanmedian(np.array(span_rate_values, dtype=float))) if len(span_rate_values) else np.nan

    resp_amp = float(np.nanmean(np.array(span_amp_values, dtype=float))) if len(span_amp_values) else np.nan
    resp_log_amp = np.log(resp_amp) if (np.isfinite(resp_amp) and resp_amp > 0) else np.nan

    return pd.DataFrame([{
        'rsa_ms': rsa_ms,
        'rsa_log_ms': rsa_log_ms,
        'resp_amp': resp_amp,
        'resp_log_amp': resp_log_amp,
        'resp_rate_bpm': resp_rate_bpm,
        'n_breaths_used': int(breaths_used_total),
        'rsa_method': 'peak_to_valley'
    }])


if __name__ == '__main__':
    print('features_segment(rr_df, resp_df=None) -> DataFrame[1 x 7] (rsa_ms, rsa_log_ms, resp_amp, resp_log_amp, resp_rate_bpm, n_breaths_used, rsa_method)')