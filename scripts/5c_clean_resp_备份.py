# scripts/4b_clean_resp.py
# 识别 resp 数据与窗口中rr数据是否长度一致，是否有缺失。保证与rr数据对齐
# 不完整的窗口 resp 数据会全部删除

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent))
        break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATASETS, ACTIVE_DATA, DATA_DIR, SCHEMA, PARAMS  # type: ignore


# ==========================================================
# I/O（必须使用你现有的 settings 路径体系）
# ==========================================================
DS = DATASETS[ACTIVE_DATA]
paths = DS["paths"]
SRC_ROOT = (DATA_DIR / paths["windowing"]).resolve()   
# 呼吸数据命名格式为 SID_resp_窗口号.csv 或 .parquet，例如 "P001S001T001R001_resp_w01.csv"
# 注意：你的原代码里用到 SRC_ROOT 但未定义，这里按项目约定改为 DATA_DIR。
SRC_DIR = (SRC_ROOT / "collected" / "resp").resolve()
# 清理后的 resp 覆盖写回同目录
OUT_ROOT = SRC_DIR

# 切窗后的窗口数据报表位置（你给的文件名与目录结构）
windowing_report = (SRC_ROOT / "collected" / "collected_index.csv").resolve()

# 报表中的窗口号列（用于 resp 文件名 wXX）
w_id_col = "final_order"
# 报表中的窗口长度列（你给的表头是 duration_s；原代码拼写错误）
w_duration_col = "duration_s"


# ==========================================================
# SID 白名单：为空则处理全部被试
# ==========================================================
SID: List[str] = [
    "P001S001T001R001",
    "P002S001T002R001",
    "P003S001T001R001",
    "P004S001T002R001",
    "P006S001T002R001",
    "P007S001T001R001",
    "P008S001T002R001",
    "P009S001T001R001",
    "P010S001T002R001",
    "P011S001T001R001",
    "P012S001T001R001",
    "P013S001T002R001",
    "P014S001T001R001",
    "P015S001T002R001",
    "P016S001T001R001",
    "P017S001T001R001",
    "P018S001T001R001",
    "P019S001T001R001",
    "P020S001T001R001",
    "P021S001T001R001",
    "P022S001T001R001",
    "P023S001T001R001",
    "P024S001T002R001",
    "P025S001T002R001",
    "P026S001T002R001",
    "P027S001T002R001",
    "P028S001T002R001",
    "P029S001T002R001",
    "P030S001T002R001",
    "P031S001T002R001",
    "P032S001T001R001",
    "P033S001T001R001",
    "P034S001T001R001",
    "P035S001T002R001",
    "P036S001T002R001",
    "P037S001T002R001",
    "P038S001T001R001",
]


# ==========================================================
# 清理策略开关（两种方法都实现，使用者可切换）
# ==========================================================
# "longest": 仅保留最长连续有效片段（推荐，最稳健，便于后续 RSA 单指标计算）
# "multi"  : 保留所有通过判定的连续有效片段（其余置 NaN；后续 RSA 计算需额外聚合策略）
CLEAN_MODE = str(PARAMS.get("resp_clean_mode", "multi"))


# ==========================================================
# 规则参数（默认值；可在 settings.PARAMS 里覆盖）
# ==========================================================
RULES: Dict[str, float] = {
    # 生理范围
    "resp_min_bpm": float(PARAMS.get("resp_min_bpm", 6.0)),
    "resp_max_bpm": float(PARAMS.get("resp_max_bpm", 30.0)),

    # Level A：时间轴
    "gap_dt_mult": float(PARAMS.get("resp_gap_dt_mult", 3.0)),
    "max_jitter": float(PARAMS.get("resp_max_jitter", 0.05)),

    # Level A：毛刺
    "spike_k": float(PARAMS.get("resp_spike_k", 9.0)),
    "spike_pad_pts": int(PARAMS.get("resp_spike_pad_pts", 1)),
    "max_spike_frac": float(PARAMS.get("resp_max_spike_frac", 0.01)),

    # Level A：饱和/夹紧（如未知可不设置）
    "resp_min_code": float(PARAMS.get("resp_min_code", np.nan)),
    "resp_max_code": float(PARAMS.get("resp_max_code", np.nan)),
    "max_sat_frac": float(PARAMS.get("resp_max_sat_frac", 0.02)),
    "max_run_sat": int(PARAMS.get("resp_max_run_sat", 80)),

    # Level A：量化/锁死
    "min_unique_ratio": float(PARAMS.get("resp_min_unique_ratio", 0.02)),
    "max_run_quantized": int(PARAMS.get("resp_max_run_quantized", 80)),
    "low_amp_iqr_abs": float(PARAMS.get("resp_low_amp_iqr_abs", 2.0)),
    # Level A：直线/近直线（整体平坦）
    "flat_iqr_abs": float(PARAMS.get("resp_flat_iqr_abs", PARAMS.get("resp_low_amp_iqr_abs", 2.0))),
    "flat_ptp_abs": float(PARAMS.get("resp_flat_ptp_abs", 5.0)),

    # 漏洞修补：局部幅度塌陷（不靠饱和阈值）
    "amp_collapse_win_sec": float(PARAMS.get("resp_amp_collapse_win_sec", 2.0)),
    "amp_collapse_frac": float(PARAMS.get("resp_amp_collapse_frac", 0.25)),
    "amp_collapse_min_run_sec": float(PARAMS.get("resp_amp_collapse_min_run_sec", 2.0)),

    # Level B：频域结构
    "bp_low_hz": float(PARAMS.get("resp_bp_low_hz", 0.10)),
    "bp_high_hz": float(PARAMS.get("resp_bp_high_hz", 0.50)),
    "min_peak_ratio": float(PARAMS.get("resp_min_peak_ratio", 0.25)),

    # Level B：周期结构
    "min_valid_sec": float(PARAMS.get("resp_min_valid_sec", 12.0)),
    "min_cycles": int(PARAMS.get("resp_min_cycles", 2)),
    "max_cv_period": float(PARAMS.get("resp_max_cv_period", 0.35)),

    # Level B：峰检鲁棒性
    "peak_prom_frac_iqr": float(PARAMS.get("resp_peak_prom_frac_iqr", 0.10)),
    "min_peak_dist_margin": float(PARAMS.get("resp_min_peak_dist_margin", 0.80)),

    # Level B：幅度下限
    "amp_min_abs": float(PARAMS.get("resp_amp_min_abs", 0.0)),
    "amp_min_frac_iqr": float(PARAMS.get("resp_amp_min_frac_iqr", 0.15)),

    # —— 按“呼吸周期”粒度做工程判定（优先保留：只删坏周期，不删整窗）——
    # 平台占比：一个周期内（diff==0）的比例，>= 0.5 认为该周期被锁死/夹紧严重
    "cycle_max_platform_frac": float(PARAMS.get("resp_cycle_max_platform_frac", 0.50)),
    # 贴边占比：一个周期内落在 ADC 最小/最大码值（0/1023）上的比例，>= 0.5 认为该周期饱和/夹紧严重
    "cycle_max_sat_frac": float(PARAMS.get("resp_cycle_max_sat_frac", 0.50)),
    # 周期内唯一值比例太低也认为量化/锁死严重
    "cycle_min_unique_ratio": float(PARAMS.get("resp_cycle_min_unique_ratio", 0.02)),
    # 周期内最长连续相同值长度（点数）超过阈值也认为锁死严重
    "cycle_max_run_quantized": int(PARAMS.get("resp_cycle_max_run_quantized", int(PARAMS.get("resp_max_run_quantized", 80)))),
    # 周期内最小振幅（原始码值幅度），低于则该周期视为无效
    "cycle_min_amp_abs": float(PARAMS.get("resp_cycle_min_amp_abs", 0.0)),

    # multi 模式可选：保留比例过低则整窗 reject（默认关闭）
    "min_kept_frac_to_accept": float(PARAMS.get("resp_min_kept_frac_to_accept", 0.0)),
}


# ==========================================================
# 工具函数
# ==========================================================


# ===== 新增：周期峰检测与周期粒度质量判定工具 =====
def _estimate_fs_from_t(t: np.ndarray) -> float:
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    return float(1.0 / dt)


def _pick_resp_peaks(t: np.ndarray, xb: np.ndarray, dt: float) -> np.ndarray:
    """在带通后的 xb 上选取能代表呼吸周期的峰（正峰或负峰择优）。"""
    resp_min_bpm = float(RULES["resp_min_bpm"])
    resp_max_bpm = float(RULES["resp_max_bpm"])

    # prominence 用 IQR 的一部分，避免噪声假峰
    q75, q25 = np.percentile(xb[np.isfinite(xb)], [75, 25])
    iqr = float(q75 - q25)
    prom = float(RULES["peak_prom_frac_iqr"] * iqr) if np.isfinite(iqr) and iqr > 0 else 0.0

    # distance：至少要大于最短周期的一部分，避免一周期内被抓出多个峰
    min_period = 60.0 / resp_max_bpm
    distance_pts = max(1, int(np.floor((min_period / dt) * float(RULES["min_peak_dist_margin"]))))

    ppos, _ = find_peaks(xb, distance=distance_pts, prominence=prom)
    pneg, _ = find_peaks(-xb, distance=distance_pts, prominence=prom)

    def _filter_by_period(peaks: np.ndarray) -> np.ndarray:
        if peaks.size < 2:
            return np.array([], dtype=int)
        keep = [int(peaks[0])]
        for i in range(peaks.size - 1):
            p0 = int(peaks[i])
            p1 = int(peaks[i + 1])
            per = float(t[p1] - t[p0])
            if not np.isfinite(per) or per <= 0:
                continue
            bpm = 60.0 / per
            if resp_min_bpm <= bpm <= resp_max_bpm:
                keep.append(p1)
        return np.array(sorted(set(keep)), dtype=int)

    vpos = _filter_by_period(ppos)
    vneg = _filter_by_period(pneg)
    return vpos if (vpos.size >= vneg.size) else vneg


def _cycle_quality_mask(t: np.ndarray, x_raw: np.ndarray, peaks: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """对每个周期（相邻 peaks 之间）做工程质量判定，返回 keep_mask（点级别）以及周期统计指标。"""
    n = x_raw.size
    keep = np.ones(n, dtype=bool)

    metrics: Dict[str, float] = {
        "cycle_n": float(max(0, peaks.size - 1)),
        "cycle_bad": 0.0,
        "cycle_bad_sat": 0.0,
        "cycle_bad_platform": 0.0,
        "cycle_bad_quant": 0.0,
        "cycle_bad_amp": 0.0,
    }

    if peaks.size < 2:
        return keep, metrics

    # 用于饱和/夹紧的码值范围（若 settings 提供就用，否则不做贴边判断）
    min_code = RULES.get("resp_min_code", np.nan)
    max_code = RULES.get("resp_max_code", np.nan)
    use_sat = np.isfinite(min_code) and np.isfinite(max_code)

    for i in range(peaks.size - 1):
        p0, p1 = int(peaks[i]), int(peaks[i + 1])
        if p1 <= p0:
            continue
        seg = x_raw[p0:p1 + 1].astype(float)
        finite = np.isfinite(seg)
        if finite.sum() < 4:
            keep[p0:p1 + 1] = False
            metrics["cycle_bad"] += 1.0
            continue

        segf = seg[finite]
        L = float(segf.size)

        # 1) 平台占比：diff==0 的比例（在原始码值上更能体现锁死/量化）
        dif = np.diff(segf)
        platform_frac = float(np.mean(dif == 0)) if dif.size > 0 else 1.0

        # 2) 贴边占比：落在 min/max code 的比例
        sat_frac = 0.0
        if use_sat:
            sat_frac = float(np.mean((segf == float(min_code)) | (segf == float(max_code))))

        # 3) 量化/锁死：唯一值比例 + 最长连续相同值
        uniq = int(np.unique(segf).size)
        unique_ratio = float(uniq / max(1, segf.size))
        # 最长连续相同值（点数）
        max_run = 1
        cur = 1
        for v in (dif == 0):
            if v:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 1

        # 4) 振幅：峰谷差
        amp = float(np.nanmax(segf) - np.nanmin(segf))

        bad_sat = (sat_frac >= float(RULES["cycle_max_sat_frac"])) if use_sat else False
        bad_platform = (platform_frac >= float(RULES["cycle_max_platform_frac"]))
        bad_quant = (unique_ratio < float(RULES["cycle_min_unique_ratio"])) or (max_run > int(RULES["cycle_max_run_quantized"]))
        amp_thr = max(float(RULES.get("cycle_min_amp_abs", 0.0)), float(RULES.get("cycle_min_amp_abs", 0.0)))
        bad_amp = (not np.isfinite(amp)) or (amp < amp_thr)

        bad = bad_sat or bad_platform or bad_quant or bad_amp
        if bad:
            keep[p0:p1 + 1] = False
            metrics["cycle_bad"] += 1.0
            if bad_sat:
                metrics["cycle_bad_sat"] += 1.0
            if bad_platform:
                metrics["cycle_bad_platform"] += 1.0
            if bad_quant:
                metrics["cycle_bad_quant"] += 1.0
            if bad_amp:
                metrics["cycle_bad_amp"] += 1.0

    return keep, metrics

def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _contiguous_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    spans: List[Tuple[int, int]] = []
    s = int(idx[0])
    p = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k == p + 1:
            p = k
        else:
            spans.append((s, p))
            s = k
            p = k
    spans.append((s, p))
    return spans


def _spans_to_time(spans: List[Tuple[int, int]], t: np.ndarray) -> List[Tuple[float, float]]:
    return [(float(t[i0]), float(t[i1])) for (i0, i1) in spans]


def _check_time_axis(t: np.ndarray) -> Tuple[bool, Dict[str, float], List[str]]:
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    if t.size < 3:
        return False, {"n": float(t.size)}, ["too_short"]

    dt = np.diff(t)
    if not np.all(np.isfinite(dt)):
        return False, {"n": float(t.size)}, ["non_finite_time"]

    if np.any(dt <= 0):
        reasons.append("time_not_strictly_increasing")
        metrics["dt_min"] = float(np.min(dt))
        return False, metrics, reasons

    med_dt = float(np.median(dt))
    jitter = float(_mad(dt) / med_dt) if med_dt > 0 else np.nan
    gap_thr = float(RULES["gap_dt_mult"] * med_dt)
    gap_frac = float(np.mean(dt > gap_thr))

    metrics.update({
        "median_dt": med_dt,
        "jitter": jitter,
        "gap_thr": gap_thr,
        "gap_frac": gap_frac,
    })

    if np.isfinite(jitter) and jitter > float(RULES["max_jitter"]):
        reasons.append("high_jitter")

    return True, metrics, reasons


def _detect_spikes(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    x0 = x.astype(float)
    finite = np.isfinite(x0)
    if finite.sum() < 4:
        return np.zeros_like(x0, dtype=bool), metrics, ["too_few_finite"]

    idx = np.where(finite)[0]
    xf = x0[finite]
    dxf = np.diff(xf)
    if dxf.size == 0:
        return np.zeros_like(x0, dtype=bool), metrics, reasons

    s = 1.4826 * _mad(dxf)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(dxf))

    k = float(RULES["spike_k"])
    spike_f = np.abs(dxf) > (k * s if np.isfinite(s) and s > 0 else np.inf)
    spike_frac = float(np.mean(spike_f)) if dxf.size > 0 else 0.0
    metrics.update({"spike_scale": float(s), "spike_frac": spike_frac})

    if np.isfinite(spike_frac) and spike_frac > float(RULES["max_spike_frac"]):
        reasons.append("many_spikes")

    spike_points = np.zeros_like(x0, dtype=bool)
    bad_next = idx[1:][spike_f]
    spike_points[bad_next] = True

    pad = int(RULES["spike_pad_pts"])
    if pad > 0 and bad_next.size > 0:
        for j in bad_next:
            lo = max(0, int(j) - pad)
            hi = min(x0.size, int(j) + pad + 1)
            spike_points[lo:hi] = True

    return spike_points, metrics, reasons


def _runs_of_equal(x: np.ndarray) -> Tuple[int, int, float]:
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return 0, 0, 0.0
    uniq = int(np.unique(xf).size)
    unique_ratio = float(uniq / max(1, xf.size))

    dif = np.diff(xf)
    if dif.size == 0:
        return int(xf.size), uniq, unique_ratio

    eq = (dif == 0)
    max_run = 1
    cur = 1
    for v in eq:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return int(max_run), uniq, unique_ratio


def _detect_quantized_or_lock(x: np.ndarray) -> Tuple[Dict[str, float], List[str]]:
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    xf = x[np.isfinite(x)].astype(float)
    if xf.size == 0:
        return metrics, ["all_nan"]

    q75, q25 = np.percentile(xf, [75, 25])
    iqr = float(q75 - q25)
    max_run, n_unique, unique_ratio = _runs_of_equal(x)

    metrics.update({
        "iqr": iqr,
        "max_run": float(max_run),
        "n_unique": float(n_unique),
        "unique_ratio": unique_ratio,
    })

    if unique_ratio < float(RULES["min_unique_ratio"]) or max_run > float(RULES["max_run_quantized"]):
        reasons.append("quantized")

    if iqr < float(RULES["low_amp_iqr_abs"]):
        reasons.append("low_amplitude")

    return metrics, reasons


def _detect_saturation(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    x0 = x.astype(float)
    finite = np.isfinite(x0)
    if finite.sum() == 0:
        return np.zeros_like(x0, dtype=bool), metrics, ["all_nan"]

    min_code = RULES.get("resp_min_code", np.nan)
    max_code = RULES.get("resp_max_code", np.nan)

    xmin = float(np.nanmin(x0))
    xmax = float(np.nanmax(x0))
    use_min = xmin if not np.isfinite(min_code) else float(min_code)
    use_max = xmax if not np.isfinite(max_code) else float(max_code)

    is_min = finite & (x0 == use_min)
    is_max = finite & (x0 == use_max)
    sat_mask = is_min | is_max

    sat_frac = float(np.mean(sat_mask[finite]))

    max_run = 0
    cur = 0
    for v in sat_mask.astype(int):
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    metrics.update({
        "sat_frac": sat_frac,
        "max_run_sat": float(max_run),
        "sat_min_code": float(use_min),
        "sat_max_code": float(use_max),
    })

    if sat_frac > float(RULES["max_sat_frac"]) or max_run > float(RULES["max_run_sat"]):
        reasons.append("saturation_or_clipping")

    return sat_mask, metrics, reasons


def _detect_amplitude_collapse(t: np.ndarray, x: np.ndarray, med_dt: float) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    """局部幅度塌陷：rolling ptp 很小且持续足够长 => 标记无效。"""
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    if t.size < 5 or not np.isfinite(med_dt) or med_dt <= 0:
        return np.zeros_like(x, dtype=bool), metrics, reasons

    win_sec = float(RULES["amp_collapse_win_sec"])
    win_pts = int(max(5, round(win_sec / med_dt)))
    if win_pts >= t.size:
        return np.zeros_like(x, dtype=bool), metrics, reasons

    xf = x.astype(float)
    finite = np.isfinite(xf)

    ptp = np.full_like(xf, np.nan, dtype=float)
    half = win_pts // 2
    for i in range(t.size):
        lo = max(0, i - half)
        hi = min(t.size, i + half + 1)
        seg = xf[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size >= max(3, win_pts // 3):
            ptp[i] = float(np.max(seg) - np.min(seg))

    med_ptp = float(np.nanmedian(ptp)) if np.isfinite(ptp).any() else np.nan
    metrics["roll_ptp_med"] = med_ptp

    # 既做“相对塌陷”（相对窗口内典型幅度），也做“绝对低幅度”（近直线/近锁死）。
    frac = float(RULES["amp_collapse_frac"])
    thr_rel = (frac * med_ptp) if (np.isfinite(med_ptp) and med_ptp > 0) else np.nan
    thr_abs = float(RULES.get("flat_ptp_abs", 5.0))

    low_rel = (ptp < thr_rel) if np.isfinite(thr_rel) else np.zeros_like(finite, dtype=bool)
    low_abs = (ptp < thr_abs)
    low = (low_rel | low_abs) & finite

    min_run_sec = float(RULES["amp_collapse_min_run_sec"])
    min_run_pts = int(max(1, round(min_run_sec / med_dt)))

    low_spans = _contiguous_spans(low)
    bad = np.zeros_like(low, dtype=bool)
    n_runs = 0
    for i0, i1 in low_spans:
        if (i1 - i0 + 1) >= min_run_pts:
            bad[i0:i1 + 1] = True
            n_runs += 1

    bad_frac = float(np.mean(bad[finite])) if finite.any() else 0.0
    metrics["amp_collapse_bad_frac"] = bad_frac
    metrics["amp_collapse_runs"] = float(n_runs)

    if n_runs > 0:
        reasons.append("amplitude_collapse_run")

    return bad, metrics, reasons


def _butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 2):
    nyq = 0.5 * fs
    low = max(1e-6, low_hz / nyq)
    high = min(0.999999, high_hz / nyq)
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid bandpass: low={low_hz}, high={high_hz}, fs={fs}")
    return butter(order, [low, high], btype="band")


def _split_into_candidate_segments(t: np.ndarray, valid_point_mask: np.ndarray, gap_thr: float) -> List[Tuple[int, int]]:
    n = t.size
    if n == 0:
        return []

    dt = np.diff(t)
    gap_break = np.zeros(n, dtype=bool)
    if dt.size > 0:
        gap_break[1:] = dt > gap_thr

    segs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        while i < n and (not valid_point_mask[i]):
            i += 1
        if i >= n:
            break
        i0 = i
        i += 1
        while i < n:
            if gap_break[i] or (not valid_point_mask[i]):
                break
            i += 1
        i1 = i - 1
        if i1 >= i0:
            segs.append((int(i0), int(i1)))
    return segs



@dataclass
class Segment:
    i0: int
    i1: int  # inclusive
    t0: float
    t1: float
    dur: float
    n_cycles: int
    f_peak: float
    peak_ratio: float
    cv_period: float
    amp_med: float


def _segment_level_b(t: np.ndarray, x: np.ndarray) -> Tuple[bool, Optional[Segment], List[str], np.ndarray]:
    """Level B：先判定这段是否具备‘像呼吸’的周期结构；返回 ok/Segment/reasons/peaks。

    约定：
    - reasons 只记录“失败原因”（会导致该段不可作为候选片段）。
    - 频谱不够尖锐等问题不作为失败，只作为后续报告可见的指标（在 metrics 中体现）。
    """
    reasons: List[str] = []

    if t.size < 4 or np.isfinite(x).sum() < 4:
        return False, None, ["too_short"], np.array([], dtype=int)

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return False, None, ["bad_dt"], np.array([], dtype=int)
    fs = float(1.0 / dt)

    low = float(RULES["bp_low_hz"])
    high = float(RULES["bp_high_hz"])

    try:
        b, a = _butter_bandpass(low, high, fs, order=2)
        xb = filtfilt(b, a, x.astype(float))
    except Exception:
        return False, None, ["bandpass_failed"], np.array([], dtype=int)

    # 频域主峰（宽松：只要求峰在带宽内且带内能量非零）
    nperseg = int(min(len(xb), max(64, int(fs * 12))))
    nperseg = max(16, nperseg)
    if nperseg >= len(xb):
        nperseg = max(16, len(xb) // 2)
    if nperseg < 16:
        return False, None, ["too_short_for_psd"], np.array([], dtype=int)

    f, pxx = welch(xb, fs=fs, nperseg=nperseg, detrend="constant")
    band = (f >= low) & (f <= high)
    if band.sum() < 3:
        return False, None, ["no_band_bins"], np.array([], dtype=int)

    fb = f[band]
    pb = pxx[band]
    band_mean = float(np.mean(pb))
    if not np.isfinite(band_mean) or band_mean <= 0:
        return False, None, ["weak_band_power"], np.array([], dtype=int)

    ip = int(np.argmax(pb))
    f_peak = float(fb[ip])
    peak_power = float(pb[ip])
    peak_ratio = float(peak_power / (band_mean + 1e-12))

    # 时域周期（峰检）
    peaks = _pick_resp_peaks(t=t, xb=xb, dt=dt)
    n_cycles = int(max(0, peaks.size - 1))
    if n_cycles < int(RULES["min_cycles"]):
        return False, None, ["too_few_cycles"], peaks

    # 周期稳定性
    cv_period = np.nan
    periods = np.diff(t[peaks]) if peaks.size >= 2 else np.array([])
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if periods.size >= 2:
        cv_period = float(np.std(periods) / (np.mean(periods) + 1e-12))
        if cv_period > float(RULES["max_cv_period"]):
            return False, None, ["unstable_period"], peaks

    # 互证：频域主峰与时域中位周期应该大致一致（允许偏差）
    if periods.size >= 1:
        f_time = float(1.0 / (np.median(periods) + 1e-12))
        # 允许 25% 相对误差 或 0.05Hz 绝对误差（两者取其一）
        if (abs(f_time - f_peak) > 0.05) and (abs(f_time - f_peak) / max(1e-6, f_peak) > 0.25):
            return False, None, ["freq_mismatch"], peaks

    dur = float(t[-1] - t[0])
    if dur < float(RULES["min_valid_sec"]):
        return False, None, ["segment_too_short"], peaks

    # 记录幅度（用于报表，不作为硬失败）
    amps: List[float] = []
    for i in range(peaks.size - 1):
        p0, p1 = int(peaks[i]), int(peaks[i + 1])
        seg = x[p0:p1 + 1]
        if seg.size == 0 or (not np.isfinite(seg).any()):
            continue
        amps.append(float(np.nanmax(seg) - np.nanmin(seg)))
    amp_med = float(np.nanmedian(amps)) if len(amps) > 0 else np.nan

    seg_obj = Segment(
        i0=0,
        i1=int(t.size - 1),
        t0=float(t[0]),
        t1=float(t[-1]),
        dur=dur,
        n_cycles=n_cycles,
        f_peak=f_peak,
        peak_ratio=peak_ratio,
        cv_period=float(cv_period) if np.isfinite(cv_period) else np.nan,
        amp_med=float(amp_med) if np.isfinite(amp_med) else np.nan,
    )
    return True, seg_obj, [], peaks


def clean_window_resp(df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, str, List[str], List[Tuple[float, float]], List[Tuple[float, float]], Dict[str, float]]:
    """单窗清理：返回 cleaned_df/status/reasons/kept_spans/dropped_spans/metrics。"""

    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    if df is None or df.empty:
        return df, "reject", ["empty"], [], [], metrics

    if df.shape[1] < 2:
        out = df.copy()
        return out, "reject", ["missing_columns"], [], [], metrics

    # 只看前两列 time_s | value（其余列保持不变）
    t = df.iloc[:, 0].to_numpy(dtype=float)
    x = df.iloc[:, 1].to_numpy(dtype=float)

    ok_time, tm, tr = _check_time_axis(t)
    metrics.update({f"time_{k}": float(v) for k, v in tm.items() if np.isfinite(v)})
    reasons.extend(tr)

    finite = np.isfinite(t) & np.isfinite(x)

    if not ok_time:
        out = df.copy()
        y = out.iloc[:, 1].to_numpy(dtype=float)
        y[:] = np.nan
        out.iloc[:, 1] = y
        return out, "reject", sorted(set(reasons)), [], _spans_to_time(_contiguous_spans(finite), t), metrics

    med_dt = float(tm["median_dt"])
    gap_thr = float(tm["gap_thr"])

    # Step0（工程底线）：本版本不因 spike/sat/quant/collapse 直接删整窗；只记录统计指标，后续按“周期粒度”删波。
    spike_mask, sm, sr = _detect_spikes(x)
    sat_mask, satm, satr = _detect_saturation(x)
    qm, qr = _detect_quantized_or_lock(x)
    collapse_mask, cm, cr = _detect_amplitude_collapse(t, x, med_dt=med_dt)

    metrics.update({f"spike_{k}": float(v) for k, v in sm.items() if np.isfinite(v)})
    metrics.update({f"sat_{k}": float(v) for k, v in satm.items() if np.isfinite(v)})
    metrics.update({f"quant_{k}": float(v) for k, v in qm.items() if np.isfinite(v)})
    metrics.update({f"collapse_{k}": float(v) for k, v in cm.items() if np.isfinite(v)})

    # 这些标签改为“存在风险的提示”，不作为 reject 的直接原因
    # 仅当最终导致无可用周期/片段时，才会在后面追加 no_valid_span 等硬失败理由
    reasons.extend(sorted(set(sr + satr + qr + cr)))

    # —— Level A 早停：整体近直线（flatline）——
    # 这类窗即使 bandpass/PSD 可能产生数值噪声，也不应被当作“可用呼吸”。
    xf = x[np.isfinite(x)].astype(float)
    if xf.size >= 10:
        q75, q25 = np.percentile(xf, [75, 25])
        iqr_all = float(q75 - q25)
        ptp_all = float(np.max(xf) - np.min(xf))
        metrics["flat_iqr_all"] = iqr_all
        metrics["flat_ptp_all"] = ptp_all
        if (iqr_all < float(RULES.get("flat_iqr_abs", RULES["low_amp_iqr_abs"]))) and (ptp_all < float(RULES.get("flat_ptp_abs", 5.0))):
            out = df.copy()
            y = out.iloc[:, 1].to_numpy(dtype=float)
            y[:] = np.nan
            out.iloc[:, 1] = y
            reasons.append("flatline_window")
            return out, "reject", sorted(set(reasons)), [], _spans_to_time(_contiguous_spans(finite), t), metrics

    # Step0：只按时间缺口与 NaN 切段（避免“工程指标误伤真实呼吸边缘”）
    valid_points = finite
    cand = _split_into_candidate_segments(t, valid_points, gap_thr=gap_thr)
    metrics["n_candidate_segments"] = float(len(cand))

    good: List[Tuple[int, int, Segment, np.ndarray]] = []
    b_fail: Dict[str, int] = {}

    for i0, i1 in cand:
        tt = t[i0:i1 + 1]
        xx = x[i0:i1 + 1]
        ok_b, seg_obj, b_reasons, peaks = _segment_level_b(tt, xx)
        if ok_b and seg_obj is not None:
            seg_obj.i0 = int(i0)
            seg_obj.i1 = int(i1)
            seg_obj.t0 = float(tt[0])
            seg_obj.t1 = float(tt[-1])
            seg_obj.dur = float(tt[-1] - tt[0])
            good.append((i0, i1, seg_obj, peaks))
        else:
            for tag in b_reasons:
                b_fail[tag] = b_fail.get(tag, 0) + 1

    for k, v in b_fail.items():
        metrics[f"B_fail_{k}"] = float(v)

    if len(good) == 0:
        out = df.copy()
        y = out.iloc[:, 1].to_numpy(dtype=float)
        y[:] = np.nan
        out.iloc[:, 1] = y
        reasons.append("no_valid_span")
        return out, "reject", sorted(set(reasons)), [], _spans_to_time(_contiguous_spans(finite), t), metrics

    # Step2：按周期删波（只删坏周期，不删整段）
    keep_mask = np.zeros_like(x, dtype=bool)
    kept_spans_idx: List[Tuple[int, int]] = []

    # 为了统计 kept_cycles/kept_dur_s，需要记录每段的“有效周期数”
    total_kept_cycles = 0

    # 先把所有 good 段都标为候选保留，再在段内按周期剔除
    for i0, i1, seg, peaks_local in good:
        # peaks_local 是相对段起点的索引；转换成全局索引
        peaks_global = (peaks_local + int(i0)).astype(int)
        # 对该段做周期质量筛查
        seg_keep, cyc_m = _cycle_quality_mask(t=t[i0:i1 + 1], x_raw=x[i0:i1 + 1], peaks=peaks_local)
        # seg_keep 是段内点级 mask（长度 = 段长），映射回全局
        keep_mask[i0:i1 + 1] |= seg_keep

        # 统计：有效周期数 = 段内周期数 - 段内坏周期数
        cyc_n = int(cyc_m.get("cycle_n", 0.0))
        cyc_bad = int(cyc_m.get("cycle_bad", 0.0))
        total_kept_cycles += max(0, (cyc_n - cyc_bad))

        # 把周期统计写到 metrics（累加）
        for kk in ["cycle_n", "cycle_bad", "cycle_bad_sat", "cycle_bad_platform", "cycle_bad_quant", "cycle_bad_amp"]:
            metrics[kk] = float(metrics.get(kk, 0.0) + float(cyc_m.get(kk, 0.0)))

    # multi/longest：在点级 keep_mask 上决定保留哪些连续片段
    keep_spans_idx = _contiguous_spans(keep_mask & finite)

    if mode not in ("longest", "multi"):
        mode = "multi"

    if mode == "longest" and len(keep_spans_idx) > 0:
        # 仅保留最长连续片段
        lens = [((j1 - j0 + 1), (j0, j1)) for (j0, j1) in keep_spans_idx]
        _, (b0, b1) = sorted(lens, key=lambda z: z[0], reverse=True)[0]
        keep_mask[:] = False
        keep_mask[b0:b1 + 1] = True
        keep_spans_idx = [(b0, b1)]

    # 重新计算 kept_frac
    kept_frac = float(np.mean(keep_mask[finite])) if finite.any() else 0.0
    metrics["kept_frac"] = kept_frac
    metrics["kept_segments"] = float(len(keep_spans_idx))
    metrics["kept_cycles"] = float(total_kept_cycles)

    # kept_dur_s：把每个 span 的时间长度加总
    kept_dur = 0.0
    for j0, j1 in keep_spans_idx:
        if j1 > j0 and np.isfinite(t[j1]) and np.isfinite(t[j0]):
            kept_dur += float(t[j1] - t[j0])
    metrics["kept_dur_s"] = float(kept_dur)

    # Step3：保留后复核（硬门槛）
    if float(metrics.get("kept_dur_s", 0.0)) < float(RULES["min_valid_sec"]):
        out = df.copy()
        y = out.iloc[:, 1].to_numpy(dtype=float)
        y[:] = np.nan
        out.iloc[:, 1] = y
        reasons.append("no_valid_span")
        reasons.append("kept_too_short")
        return out, "reject", sorted(set(reasons)), [], _spans_to_time(_contiguous_spans(finite), t), metrics

    if float(metrics.get("kept_cycles", 0.0)) < float(RULES["min_cycles"]):
        out = df.copy()
        y = out.iloc[:, 1].to_numpy(dtype=float)
        y[:] = np.nan
        out.iloc[:, 1] = y
        reasons.append("no_valid_span")
        reasons.append("too_few_cycles")
        return out, "reject", sorted(set(reasons)), [], _spans_to_time(_contiguous_spans(finite), t), metrics

    min_kept_frac = float(RULES.get("min_kept_frac_to_accept", 0.0))
    if min_kept_frac > 0 and kept_frac < min_kept_frac:
        out = df.copy()
        y = out.iloc[:, 1].to_numpy(dtype=float)
        y[:] = np.nan
        out.iloc[:, 1] = y
        reasons.append("kept_frac_too_low")
        return out, "reject", sorted(set(reasons)), [], _spans_to_time(_contiguous_spans(finite), t), metrics

    out = df.copy()
    y = out.iloc[:, 1].to_numpy(dtype=float)
    y[~keep_mask] = np.nan
    out.iloc[:, 1] = y

    changed = bool(np.any(finite & (~keep_mask)))
    status = "partial" if changed else "ok"
    if status == "partial":
        reasons.append("trimmed")

    kept_spans = _spans_to_time(keep_spans_idx, t)
    dropped_mask = finite & (~keep_mask)
    dropped_spans = _spans_to_time(_contiguous_spans(dropped_mask), t)

    return out, status, sorted(set(reasons)), kept_spans, dropped_spans, metrics


# ==========================================================
# I/O helpers（保持原始文件格式：csv / parquet）
# ==========================================================

def _read_table(fp: Path) -> pd.DataFrame:
    suf = fp.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(fp)
    return pd.read_csv(fp)


def _write_table(df: pd.DataFrame, fp: Path) -> None:
    suf = fp.suffix.lower()
    if suf in (".parquet", ".pq"):
        df.to_parquet(fp, index=False)
    else:
        df.to_csv(fp, index=False)


def _find_resp_file(sid: str, w: int) -> Optional[Path]:
    # 优先 parquet，其次 csv
    stem = f"{sid}_resp_w{w:02d}"
    cand = [
        SRC_DIR / f"{stem}.parquet",
        SRC_DIR / f"{stem}.pq",
        SRC_DIR / f"{stem}.csv",
    ]
    for fp in cand:
        if fp.exists():
            return fp
    return None


# ==========================================================
# 主流程（VSCode 直接 Run）
# ==========================================================

def main() -> None:
    if not windowing_report.exists():
        raise FileNotFoundError(f"windowing_report not found: {windowing_report}")

    rep = pd.read_csv(windowing_report)

    # 校验列名（你给的 collected_index.csv 头部）
    required_cols = {"subject_id", w_id_col, w_duration_col}
    missing = [c for c in required_cols if c not in rep.columns]
    if missing:
        raise ValueError(
            "collected_index.csv 列名不匹配。缺失列: " + ",".join(missing)
            + f" | 当前列: {list(rep.columns)}"
        )

    # 过滤 SID
    if len(SID) > 0:
        rep = rep[rep["subject_id"].isin(SID)].copy()

    # 仅使用 level1/level2/... 都可，因为 final_order 是全局序号，resp 文件也按它命名
    rep[w_id_col] = rep[w_id_col].astype(int)

    # 逐 SID 清理
    rows = []
    n_ok = n_partial = n_reject = 0

    for sid, g in rep.groupby("subject_id", sort=True):
        g = g.sort_values(w_id_col)
        for _, r in g.iterrows():
            w = int(r[w_id_col])
            expected_dur = float(r[w_duration_col]) if np.isfinite(r[w_duration_col]) else np.nan

            fp = _find_resp_file(sid, w)
            if fp is None:
                rows.append({
                    "subject_id": sid,
                    "final_order": w,
                    "file": "",
                    "status": "missing",
                    "reasons": "[\"file_not_found\"]",
                    "kept_spans": "[]",
                    "dropped_spans": "[]",
                    "expected_duration_s": expected_dur,
                })
                continue

            df = _read_table(fp)

            # 轻量一致性检查：若 time 列跨度明显小于报表 duration_s（比如缺一大段）
            # 注意：不直接删窗，只作为 reason 标签，真正是否保留由规则判定。
            reasons_extra: List[str] = []
            if df.shape[1] >= 2:
                t = df.iloc[:, 0].to_numpy(dtype=float)
                if np.isfinite(expected_dur) and t.size >= 2 and np.all(np.isfinite(t)):
                    span = float(t[-1] - t[0])
                    if span < 0.8 * expected_dur:
                        reasons_extra.append("shorter_than_expected")

            cleaned, status, reasons, kept_spans, dropped_spans, metrics = clean_window_resp(df, mode=CLEAN_MODE)
            if reasons_extra:
                reasons = sorted(set(reasons + reasons_extra))

            # 覆盖写回（格式保持不变）
            out_fp = OUT_ROOT / fp.name
            _write_table(cleaned, out_fp)

            if status == "ok":
                n_ok += 1
            elif status == "partial":
                n_partial += 1
            else:
                n_reject += 1

            row = {
                "subject_id": sid,
                "final_order": w,
                "file": fp.name,
                "status": status,
                "reasons": str(reasons),
                "kept_spans": str(kept_spans),
                "dropped_spans": str(dropped_spans),
                "expected_duration_s": expected_dur,
            }
            # 常用指标写入报表，方便你后续调参对照
            for k in [
                "time_median_dt", "time_jitter", "time_gap_frac",
                "spike_spike_frac",
                "sat_sat_frac", "sat_max_run_sat",
                "quant_iqr", "quant_unique_ratio", "quant_max_run",
                "collapse_roll_ptp_med", "collapse_amp_collapse_bad_frac", "collapse_amp_collapse_runs",
                "n_candidate_segments", "kept_frac", "kept_segments", "kept_dur_s", "kept_cycles",
                "flat_iqr_all", "flat_ptp_all",
            ]:
                if k in metrics:
                    row[k] = metrics[k]
            rows.append(row)

    out_report = OUT_ROOT / "resp_clean_report.csv"
    pd.DataFrame(rows).to_csv(out_report, index=False)

    print(f"[5c_clean_resp] DONE | mode={CLEAN_MODE} | ok={n_ok} partial={n_partial} reject={n_reject} | report={out_report}")


if __name__ == "__main__":
    main()