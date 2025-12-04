# scripts/4b_clean_rr.py
# 功能：读取 decision.csv 与已确认的 RR（通常为 device_rr），仅针对“连续 1–5 拍短时尖峰”做最小化清理：
#   - 识别：基于滚动中位数的相对偏差 + 逐搏差绝对阈值，聚合为 1–5 拍短段；
#   - 分类：对短段执行“补偿判据”以识别错分；
#   - 参考：可选从 ECG 快速提取 ecg_rr，若判定“两路皆异常”，只写 QC，不自动修复；
#   - 修复：普通短段→局部插值；错分短段→成对合并（可开关）；
#   - 输出：覆盖写回确认目录中的 RR（csv/parquet 若存在则一并覆盖），并保存 QC 摘要表。

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import neurokit2 as nk
# 可选：hrvanalysis 主要用于与既有口径保持一致（本脚本核心识别逻辑自行实现，更可控）
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values  # noqa: F401

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent))
        break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATASETS, ACTIVE_DATA, DATA_DIR, SCHEMA, PARAMS  # type: ignore

DS = DATASETS[ACTIVE_DATA]
paths = DS["paths"]
SRC_RR_DIR = (DATA_DIR / paths["confirmed"]).resolve()   # 确认后的 RR（将被覆盖写回）
SRC_ECG_DIR = (DATA_DIR / paths["norm"]).resolve()       # 原始/规范化 ECG 所在目录
OUT_ROOT = SRC_RR_DIR                                      # 覆盖式输出 → 与确认目录一致
SELECT_RR_REF_DIR = (DATA_DIR / paths["rr_select"]).resolve()  # 包含 decision.csv 的目录

# ----- 坏点修复策略 -----
# 对于“坏点”有三种修复策略，默认方法是“插值”（interp）处理方法，该方法删除异常点，并以插值的方法对删除的局部窗口进行平滑处理。
# 第二种方法是“切分”（split）方法。该方法针对“漏检”的情况，比如少检测一个R波，导致两个期间连在一起，这经常体现在一些突然很高的异常值上，例如从 600 突然跳到 1200。
# 该方法会将 1200 位置的R波拆分为2个600的R波。
# 
# "interp"：保留现有插值逻辑（默认）
# "split" ：先对多拍合并 RR 做切分，然后其它短尖峰仍按插值处理。
# "delete": 已识别的短段直接删除，不做插值
# "fill": 删掉一个异常点，用前后两个点的平均值替换
RR_REPAIR_METHOD  = ["split","interp"]

# ----- 多拍合并切分相关参数 -----
# 识别“多拍合并”的最小 RR（毫秒），低于这个值即便是整数倍也不认为是漏检多拍
RR_MULTI_MIN_MS   = 1200
# 认为最多可能连在一起的拍数，例如 5 表示最多允许识别为 5 拍合并
RR_MULTI_MAX_K    = 5
# rr ≈ k * median(rr) 的容差，比如 0.12 = 相对误差 < 12%
RR_MULTI_REL_TOL  = 0.12

# —— 基于插值方法的 RR 清理参数 ——
RR_REL_THR = float(PARAMS.get("rr_artifact_threshold", 0.25))
HR_MIN, HR_MAX = PARAMS.get("hr_min_max_bpm", (35, 140))
RR_LOW_MS = 510.0 # 这个值用于过滤离群值（最低），先绘图，根据图来判断阈值
RR_HIGH_MS = 2000.0 # 最高的离群值阈值
RR_DABS_MS = float(PARAMS.get("rr_delta_abs_ms", 150.0))
# RUN_MAX 是连续坏点的个数达到上限，超过次限制不再修复
RUN_MAX = int(PARAMS.get("rr_short_run_max_beats", 10))
COMP_TOL = float(PARAMS.get("rr_pair_comp_tolerance", 0.15))
GROUPS_WARN = int(PARAMS.get("rr_groups_warn", 4))
RATIO_WARN = float(PARAMS.get("rr_max_correct_ratio_warn", 0.05))
BOTH_BAD_THR = float(PARAMS.get("both_bad_rel_dev_thr", 0.15))
INTERP_METHOD = str(PARAMS.get("interp_method", "pchip")).lower()

# 禁止对两个点直接相加，当出现“一短一长”且“两者之和等于两倍正常值”时，代码会触发 BUG
# MERGE_ENABLE = bool(PARAMS.get("pair_merge_enable", True))
MERGE_ENABLE = False


RR_T = SCHEMA["rr"]["t"]
RR_V = SCHEMA["rr"]["v"]
ECG_T = SCHEMA["ecg"]["t"]
ECG_V = SCHEMA["ecg"]["v"]

# 处理那种 rr 数据，默认都处理
CLEAN = ["device_rr", "ecg_rr"] # ["device_rr", "ecg_rr"]

def _rolling_median_exclude_center(x: pd.Series, win: int = 11) -> pd.Series:
    """中心对齐滚动中位数；为稳健起见，使用窗口中位数近似“去中心”。"""
    if win % 2 == 0:
        win += 1
    m = x.rolling(window=win, center=True, min_periods=max(3, win // 2)).median()
    return m

def _group_runs(indices: np.ndarray) -> List[Tuple[int, int]]:
    """将升序索引中的相邻元素聚合为闭区间段 [(s,e), ...]。"""
    if indices.size == 0:
        return []
    runs: List[Tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for k in indices[1:]:
        k = int(k)
        if k == prev + 1:
            prev = k
            continue
        runs.append((start, prev))
        start = prev = k
    runs.append((start, prev))
    return runs

# 识别有问题的“坏点”
def _detect_short_spike_segments(df: pd.DataFrame) -> List[Dict]:
    """识别连续 1–RUN_MAX 拍的短时异常段，并标注“补偿判据”。"""
    rr = pd.to_numeric(df[RR_V], errors="coerce")
    t = pd.to_numeric(df[RR_T], errors="coerce")
    m = _rolling_median_exclude_center(rr, win=11)
    # 候选：相对偏差 OR 绝对 RR 超阈 OR 逐搏差绝对值超阈
    rel_dev = (rr - m).abs() / m
    diff_abs = rr.diff().abs()
    cand = (
        rel_dev > RR_REL_THR
    ) | (rr < RR_LOW_MS) | (rr > RR_HIGH_MS) | (diff_abs > RR_DABS_MS)
    idx = np.where(cand.fillna(False).to_numpy())[0]
    runs = _group_runs(idx)
    # 保留 1..RUN_MAX 的短段
    short_runs = [(s, e) for (s, e) in runs if 1 <= (e - s + 1) <= RUN_MAX]
    segments: List[Dict] = []
    for s, e in short_runs:
        length = e - s + 1
        # 补偿判据：仅在 length>=2 时检查（成对短 RR）
        compensation = False
        if length >= 2:
            ok = 0
            pairs = 0
            for i in range(s, e):
                Rsum = rr.iloc[i] + rr.iloc[i + 1]
                mloc = float(m.iloc[i]) if pd.notna(m.iloc[i]) else float(rr.iloc[i])
                if mloc <= 0:
                    continue
                pairs += 1
                dev = abs(Rsum - 2.0 * mloc) / (2.0 * mloc)
                if dev < COMP_TOL:
                    ok += 1
            if pairs > 0 and ok / pairs >= 0.5:
                compensation = True
        seg = {
            "i_start": int(s),
            "i_end": int(e),
            "n_beats": int(length),
            "t_start": float(t.iloc[s]),
            "t_end": float(t.iloc[e]),
            "compensation": compensation,
        }
        segments.append(seg)
    return segments


# --- 新增：三种检测函数，均复用短尖峰检测 ---
def _detect_segments_for_interp(df: pd.DataFrame) -> List[Dict]:
    """插值清理使用的异常段检测，目前直接复用短尖峰检测。"""
    return _detect_short_spike_segments(df)


def _detect_segments_for_delete(df: pd.DataFrame) -> List[Dict]:
    """删除清理使用的异常段检测，目前直接复用短尖峰检测。"""
    return _detect_short_spike_segments(df)


def _detect_segments_for_fill(df: pd.DataFrame) -> List[Dict]:
    """局部填补清理使用的异常段检测，目前直接复用短尖峰检测。"""
    return _detect_short_spike_segments(df)

# 针对漏检时，产生的N倍于其他拍的R值，找到这些值，并对其拆分
def _split_long_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别形如 rr ≈ k * local_median, k>=2 的“多拍合并”长 RR，
    在 [t_{i-1}, t_i] 区间内切分为 k 个等长 RR，并插入对应时间戳。

    注意：
    - 不修改原有终点 t_i，也不平移后续时间轴；
    - 若不满足判定条件，则原样返回。
    """
    if df.empty:
        return df

    rr = pd.to_numeric(df[RR_V], errors="coerce").to_numpy()
    tt = pd.to_numeric(df[RR_T], errors="coerce").to_numpy()
    n = rr.size

    # 与短尖峰检测使用相同的滚动中位数窗口
    m_series = _rolling_median_exclude_center(pd.Series(rr), win=11)
    mm = m_series.to_numpy()

    new_t: list[float] = []
    new_rr: list[float] = []

    for i in range(n):
        # 默认：原样复制
        if i == 0 or not np.isfinite(rr[i]) or not np.isfinite(mm[i]) or rr[i] < RR_MULTI_MIN_MS:
            new_t.append(tt[i])
            new_rr.append(rr[i])
            continue

        # 估计可能的合并拍数 k
        k_float = rr[i] / mm[i] if mm[i] > 0 else np.inf
        k = int(round(k_float))

        if 2 <= k <= RR_MULTI_MAX_K:
            target = k * mm[i]
            rel_dev = abs(rr[i] - target) / target if target > 0 else np.inf
        else:
            rel_dev = np.inf

        # 满足“长 RR”“k 合理”“误差在阈值内”才切分
        if (rr[i] >= RR_MULTI_MIN_MS) and (2 <= k <= RR_MULTI_MAX_K) and (rel_dev < RR_MULTI_REL_TOL):
            t_prev = tt[i - 1]
            total_dt = tt[i] - t_prev

            # 时间轴本身异常则放弃切分
            if not np.isfinite(total_dt) or total_dt <= 0:
                new_t.append(tt[i])
                new_rr.append(rr[i])
                continue

            dt = total_dt / k
            rr_piece = total_dt * 1000.0 / k  # 秒 → 毫秒，再均分

            # 在 (t_prev, t_i] 区间内填回 k 个搏点
            for j in range(1, k + 1):
                t_j = t_prev + j * dt
                new_t.append(t_j)
                new_rr.append(rr_piece)
        else:
            new_t.append(tt[i])
            new_rr.append(rr[i])

    return pd.DataFrame({RR_T: new_t, RR_V: new_rr})

def _load_confirmed_rr_path(sid: str) -> Dict[str, Path]:
    """找到确认目录下该被试的 RR 文件路径（csv/parquet 若存在均返回）。"""
    out: Dict[str, Path] = {}
    # 优先精确匹配 *_rr.csv / *_rr.parquet
    for ext in (".csv", ".parquet"):
        p = SRC_RR_DIR / f"{sid}_rr{ext}"
        if p.exists():
            out[ext] = p
    # 回退：通配匹配
    if not out:
        for ext in (".csv", ".parquet"):
            cand = list(SRC_RR_DIR.glob(f"{sid}*rr*{ext}"))
            if cand:
                # 取最短文件名（更可能是标准命名）
                cand.sort(key=lambda x: len(x.name))
                out[ext] = cand[0]
    return out

def _read_rr(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    # 兼容列名
    tcol = df.columns[0].lower()
    vcol = df.columns[1].lower()
    out = df[[tcol, vcol]].rename(columns={tcol: RR_T, vcol: RR_V}).copy()
    return out

def _write_rr(path_pref: Dict[str, Path], df: pd.DataFrame) -> None:
    # 覆盖写回：存在什么就写什么
    if ".csv" in path_pref:
        df[[RR_T, RR_V]].to_csv(path_pref[".csv"], index=False)
    if ".parquet" in path_pref:
        df[[RR_T, RR_V]].to_parquet(path_pref[".parquet"], index=False)

def _load_ecg_rr_from_norm(sid: str) -> pd.DataFrame | None:
    """从规范化目录读取 ECG，快速生成 ecg_rr（近似即可，用于“皆异常”判定）。"""
    # 1) 定位 ECG 文件
    ecg_path: Path | None = None
    for ext in (".parquet", ".csv"):
        p = SRC_ECG_DIR / f"{sid}_ecg{ext}"
        if p.exists():
            ecg_path = p
            print(f"[find ecg] {sid} 的 ECG 文件路径: {ecg_path}")
            break
    if ecg_path is None:
        # 回退通配
        cand = list(SRC_ECG_DIR.glob(f"{sid}*ecg*.*"))
        if cand:
            cand.sort(key=lambda x: len(x.name))
            ecg_path = cand[0]
    if ecg_path is None or not ecg_path.exists():
        return None
    # 2) 读取 ECG
    if ecg_path.suffix == ".csv":
        df = pd.read_csv(ecg_path)
    else:
        df = pd.read_parquet(ecg_path)
    print(f"[ecg column] {df.columns}")
    tcol = df.columns[0].lower()
    vcol = df.columns[1].lower()
    sig = df[[tcol, vcol]].rename(columns={tcol: ECG_T, vcol: ECG_V}).dropna()
    # 3) 估计 fs 并重采样到等间隔
    t = sig[ECG_T].to_numpy(dtype=float)
    v = sig[ECG_V].to_numpy(dtype=float)
    if t.size < 5:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    fs = float(np.clip(np.round(1.0 / np.median(dt)), 50, 2000))  # 保守限幅
    t_grid = np.arange(t[0], t[-1], 1.0 / fs)
    if t_grid.size < 5:
        return None
    v_grid = np.interp(t_grid, t, v)
    # 4) R 峰检测
    try:
        _, info = nk.ecg_peaks(v_grid, sampling_rate=fs)
        r_idx = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        if r_idx.size < 3:
            return None
    except Exception:
        return None
    r_t = t_grid[r_idx]
    rr_ms = np.diff(r_t) * 1000.0
    if rr_ms.size == 0:
        return None
    t_rr = r_t[1:]  # RR 记到后一个 R 峰时刻
    out = pd.DataFrame({RR_T: t_rr, RR_V: rr_ms})
    return out

def _both_bad(ecg_rr: pd.DataFrame, segs: List[Dict]) -> bool:
    if ecg_rr is None or ecg_rr.empty:
        return False
    t_ecg = ecg_rr[RR_T].to_numpy()
    rr_ecg = ecg_rr[RR_V].to_numpy()
    # 预计算滚动中位数（简化：用全局中位数近似局部基线）
    # 对于短窗判定，直接用局部中位数更合适：按片段内的中位数即可
    for s in segs:
        a = s["t_start"] - 2.0
        b = s["t_end"] + 2.0
        sel = (t_ecg >= a) & (t_ecg <= b)
        if not np.any(sel):
            continue
        rr_win = rr_ecg[sel]
        if rr_win.size < 3:
            continue
        med = np.median(rr_win)
        rel = np.abs(rr_win - med) / max(1e-6, med)
        if np.median(rel) > BOTH_BAD_THR:
            return True
    return False

def _interp_local(t: np.ndarray, y: np.ndarray, mask_nan: np.ndarray) -> np.ndarray:
    """对 y 中的 NaN 位置进行按时间的一维插值；优先 PCHIP，退化为线性；边缘位置做邻值填充。"""
    y_new = y.copy()
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return y_new
    x = t
    xk = x[valid]
    yk = y[valid]
    try:
        if INTERP_METHOD == "pchip":
            f = PchipInterpolator(xk, yk, extrapolate=False)
            y_new[mask_nan] = f(x[mask_nan])
        else:
            y_new[mask_nan] = np.interp(x[mask_nan], xk, yk)
    except Exception:
        y_new[mask_nan] = np.interp(x[mask_nan], xk, yk)
    # 边缘 NaN（若有）用最近邻填充
    if np.isnan(y_new[0]):
        first = np.flatnonzero(~np.isnan(y_new))
        if first.size:
            y_new[0] = y_new[first[0]]
    if np.isnan(y_new[-1]):
        last = np.flatnonzero(~np.isnan(y_new))
        if last.size:
            y_new[-1] = y_new[last[-1]]
    return y_new

# 针对device_rr的三种修复方式：第一种，删除法
def _repair_delete(df: pd.DataFrame, segs: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    删除法清理：对异常段内所有搏点直接删除。
    对补偿段（compensation=True 且启用 MERGE_ENABLE）仍执行成对合并。
    """
    rr = pd.to_numeric(df[RR_V], errors="coerce").to_numpy()
    tt = pd.to_numeric(df[RR_T], errors="coerce").to_numpy()
    n = rr.size

    to_drop = np.zeros(n, dtype=bool)
    records: List[Dict] = []

    for s in segs:
        i0, i1 = s["i_start"], s["i_end"]
        nlen = i1 - i0 + 1

        if s.get("compensation", False) and MERGE_ENABLE and nlen >= 2:
            j = i0
            merged_pairs = 0
            while j + 1 <= i1:
                rr[j + 1] = rr[j] + rr[j + 1]
                to_drop[j] = True
                merged_pairs += 1
                j += 2
            if (i1 - i0 + 1) % 2 == 1:
                # 奇数个，最后一个未合并，按删除策略一并删除
                to_drop[i1] = True
            records.append({
                "t_start": float(tt[i0]),
                "t_end": float(tt[i1]),
                "n_beats": nlen,
                "method": "delete_pair_merge",
                "mode": "delete",
                "merged_pairs": merged_pairs,
            })
        else:
            to_drop[i0:i1 + 1] = True
            records.append({
                "t_start": float(tt[i0]),
                "t_end": float(tt[i1]),
                "n_beats": nlen,
                "method": "delete",
                "mode": "delete",
            })

    keep = ~to_drop
    out = df.loc[keep, [RR_T, RR_V]].reset_index(drop=True)
    return out, records

# 针对device_rr的三种修复方式：第二种，时间加权填补法
def _repair_fill(df: pd.DataFrame, segs: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    局部填补法：
      - 若异常段仅包含 1 个坏点，且前后各有一个正常点，则使用“邻点时间加权均值”替换该点：
            w_prev = dt_next / (dt_prev + dt_next)
            w_next = dt_prev / (dt_prev + dt_next)
            rr[i] = w_prev * rr[i-1] + w_next * rr[i+1]
        其中 dt_prev = t_i - t_{i-1}, dt_next = t_{i+1} - t_i。
      - 对补偿段仍执行成对合并；
      - 对无法安全填补的情况（段长>1，或在边界等），回退为插值处理。
    """
    rr = pd.to_numeric(df[RR_V], errors="coerce").to_numpy()
    tt = pd.to_numeric(df[RR_T], errors="coerce").to_numpy()
    n = rr.size

    to_nan = np.zeros(n, dtype=bool)
    to_drop = np.zeros(n, dtype=bool)
    records: List[Dict] = []

    for s in segs:
        i0, i1 = s["i_start"], s["i_end"]
        nlen = i1 - i0 + 1

        # 1) 先处理补偿段
        if s.get("compensation", False) and MERGE_ENABLE and nlen >= 2:
            j = i0
            merged_pairs = 0
            while j + 1 <= i1:
                rr[j + 1] = rr[j] + rr[j + 1]
                to_drop[j] = True
                merged_pairs += 1
                j += 2
            if (i1 - i0 + 1) % 2 == 1:
                # 奇数个，最后一个未合并，交给插值
                to_nan[i1] = True
            records.append({
                "t_start": float(tt[i0]),
                "t_end": float(tt[i1]),
                "n_beats": nlen,
                "method": "pair_merge",
                "mode": "fill",
                "merged_pairs": merged_pairs,
            })
            continue

        # 2) 非补偿段：单点且前后都有点 → 时间加权填补
        if (nlen == 1) and (0 < i0 < n - 1) \
           and np.isfinite(rr[i0 - 1]) and np.isfinite(rr[i0 + 1]) \
           and np.isfinite(tt[i0 - 1]) and np.isfinite(tt[i0]) and np.isfinite(tt[i0 + 1]):

            dt_prev = tt[i0] - tt[i0 - 1]
            dt_next = tt[i0 + 1] - tt[i0]

            if dt_prev > 0 and dt_next > 0:
                denom = dt_prev + dt_next
                w_prev = dt_next / denom
                w_next = dt_prev / denom
                rr[i0] = float(w_prev * rr[i0 - 1] + w_next * rr[i0 + 1])
                records.append({
                    "t_start": float(tt[i0]),
                    "t_end": float(tt[i0]),
                    "n_beats": nlen,
                    "method": "fill_weighted",
                    "mode": "fill",
                    "w_prev": float(w_prev),
                    "w_next": float(w_next),
                })
                # 不标记 to_nan / to_drop，该点就此完成填补
                continue

        # 3) 其它情况：回退为插值处理
        to_nan[i0:i1 + 1] = True
        records.append({
            "t_start": float(tt[i0]),
            "t_end": float(tt[i1]),
            "n_beats": nlen,
            "method": "interp_fallback",
            "mode": "fill",
        })

    # 4) 对 fill 模式中标记为 NaN 的位置做一次局部插值
    keep = ~to_drop
    rr2 = rr[keep]
    tt2 = tt[keep]

    idx_map = np.full(n, -1, dtype=int)
    idx_map[keep] = np.arange(keep.sum(), dtype=int)

    to_nan_new = np.zeros(rr2.size, dtype=bool)
    src_nan_idx = np.where(to_nan)[0]
    for k in src_nan_idx:
        j = idx_map[k]
        if j >= 0:
            to_nan_new[j] = True

    rr2[to_nan_new] = np.nan
    rr_filled = _interp_local(tt2, rr2, to_nan_new)

    out = pd.DataFrame({RR_T: tt2, RR_V: rr_filled})
    return out, records

# 针对device_rr的三种修复方式：第三种，插值法
def _repair_interp(df: pd.DataFrame, segs: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    插值法清理：将异常段内搏点标记为 NaN，之后在时间轴上用 _interp_local 填补。
    对补偿段仍执行成对合并。
    """
    rr = pd.to_numeric(df[RR_V], errors="coerce").to_numpy()
    tt = pd.to_numeric(df[RR_T], errors="coerce").to_numpy()
    n = rr.size

    to_nan = np.zeros(n, dtype=bool)
    to_drop = np.zeros(n, dtype=bool)
    records: List[Dict] = []

    for s in segs:
        i0, i1 = s["i_start"], s["i_end"]
        nlen = i1 - i0 + 1

        if s.get("compensation", False) and MERGE_ENABLE and nlen >= 2:
            # 成对合并：两两相加，保留后一个，删除前一个；若为奇数，最后一个留待插值
            j = i0
            merged_pairs = 0
            while j + 1 <= i1:
                rr[j + 1] = rr[j] + rr[j + 1]
                to_drop[j] = True
                merged_pairs += 1
                j += 2
            if (i1 - i0 + 1) % 2 == 1:
                # 奇数个，最后一个未合并，改为插值
                to_nan[i1] = True
            records.append({
                "t_start": float(tt[i0]),
                "t_end": float(tt[i1]),
                "n_beats": nlen,
                "method": "pair_merge",
                "mode": "interp",
                "merged_pairs": merged_pairs,
            })
        else:
            to_nan[i0:i1 + 1] = True
            records.append({
                "t_start": float(tt[i0]),
                "t_end": float(tt[i1]),
                "n_beats": nlen,
                "method": "interp",
                "mode": "interp",
            })

    keep = ~to_drop
    rr2 = rr[keep]
    tt2 = tt[keep]

    idx_map = np.full(n, -1, dtype=int)
    idx_map[keep] = np.arange(keep.sum(), dtype=int)

    to_nan_new = np.zeros(rr2.size, dtype=bool)
    src_nan_idx = np.where(to_nan)[0]
    for k in src_nan_idx:
        j = idx_map[k]
        if j >= 0:
            to_nan_new[j] = True

    rr2[to_nan_new] = np.nan
    rr_filled = _interp_local(tt2, rr2, to_nan_new)

    out = pd.DataFrame({RR_T: tt2, RR_V: rr_filled})
    return out, records


def _apply_repairs(df: pd.DataFrame, segs: List[Dict], mode: str = "interp") -> Tuple[pd.DataFrame, List[Dict]]:
    """
    按标签对短段执行修复，具体方法由 mode 参数决定：
      - "delete"：调用 _repair_delete
      - "fill"  ：调用 _repair_fill
      - 其它    ：调用 _repair_interp
    返回修复后的 DF 与每段的处理记录。
    """
    m = mode.lower()
    if m == "delete":
        return _repair_delete(df, segs)
    elif m == "fill":
        return _repair_fill(df, segs)
    else:
        return _repair_interp(df, segs)


def clean_device_rr(sid: str, summary_records: list) -> None:
    path_pref = _load_confirmed_rr_path(sid)
    if not path_pref:
        print(f"[warn] 未在确认目录找到 {sid} 的 RR 文件，跳过。")
        return

    # 读取确认目录中的 RR
    df = None
    if ".csv" in path_pref:
        df = _read_rr(path_pref[".csv"])
    elif ".parquet" in path_pref:
        df = _read_rr(path_pref[".parquet"])

    if df is None or df.empty:
        print(f"[warn] {sid} RR 文件为空或不可读，跳过。")
        return

    df_cur = df.copy()

    # 统一初始化清理过程中的统计与标记
    warn_msgs: list[str] = []
    total_corr_beats = 0
    total_segments = 0
    both_bad_any = False

    # 可选加载 ECG RR，用于后续“两路皆异常”判定
    ecg_rr = _load_ecg_rr_from_norm(sid)

    # ---- 1) 处理方法配置，并强制 split 作为预处理 ----
    method_sequence = [m.lower() for m in RR_REPAIR_METHOD]

    # 是否需要做 split 预处理
    if "split" in method_sequence:
        before_len = len(df_cur)
        df_cur = _split_long_multiples(df_cur)
        after_len = len(df_cur)
        if after_len != before_len:
            warn_msgs.append(f"split: 调整/插入搏点 {after_len - before_len} 个")
        # 从后续步骤里移除 split，避免再次执行
        method_sequence = [m for m in method_sequence if m != "split"]

    # 其余方法按用户顺序执行（interp / fill / delete 等）
    for m in method_sequence:
        if m in ("interp", "delete", "fill"):
            # 按方法选择对应的检测函数
            seg_detector = {
                "interp": _detect_segments_for_interp,
                "delete": _detect_segments_for_delete,
                "fill":   _detect_segments_for_fill,
            }[m]
            segs = seg_detector(df_cur)
            if not segs:
                continue

            # 累积统计信息
            n_corr_beats_step = int(sum(s["n_beats"] for s in segs))
            total_corr_beats += n_corr_beats_step
            total_segments += len(segs)

            # 若提供 ECG RR，则检查“两路皆异常”
            if ecg_rr is not None and _both_bad(ecg_rr, segs):
                both_bad_any = True

            # 按当前模式执行修复
            df_cur, recs = _apply_repairs(df_cur, segs, mode=m)
            continue

        else:
            print(f"[warn] 未知 RR 清理方法 {m}，跳过。")

    # 3) 提醒条件（不中断）
    n_corr_beats = total_corr_beats
    n_segments = total_segments
    ratio_corr = n_corr_beats / max(1, len(df))

    if n_segments >= GROUPS_WARN:
        warn_msgs.append(f"异常短段数量≥{GROUPS_WARN}")
    if ratio_corr >= RATIO_WARN:
        warn_msgs.append(f"需修正搏点占比≥{RATIO_WARN:.2f}")

    if both_bad_any:
        warning_text = "; ".join(["两路皆异常"] + warn_msgs) if warn_msgs else "两路皆异常"
    else:
        warning_text = "; ".join(warn_msgs)

    # 4) 写回清理后的 RR
    _write_rr(path_pref, df_cur)

    # 5) 记录 QC 摘要
    summary_records.append({
        "subject_id": sid,
        "n_segments": n_segments,
        "n_corrected": n_corr_beats,
        "ratio_corrected": round(ratio_corr, 4),
        "warning": warning_text,
    })

def clean_ecgv2_rr(sid: str, summary_records: list) -> None:
    """
    清理 v2 算法产生的 RR（来自 ECG→RR 的逐搏表：列须含 t_s, rr_ms）。
    - 标记规则（取并集）：
        1) 生理越界：RR < 300 ms 或 RR > 2000 ms
        2) 相邻相对突变：|ΔRR| / max(prev,1e-6) > rel_thr（默认 0.25）
        3) 局部中位距：|RR - rolling_median(W=5)| / median > loc_thr（默认 0.25）
    - 将 True 连续片段合并为“短段”（长度 1..MAX_SEG_BEATS，默认 4），两侧需各有至少一个有效点。
    - 对每个短段使用“以时间为自变量”的线性插值修复（保持时间轴与 HR 1Hz 聚合一致）。
    - 写回确认目录，并把修复统计加入 summary_records。
    """
    # 读取确认目录中的 RR
    print(f'[clean ecg rr] 清理 {sid} 的 ecg_rr数据')
    path_pref = _load_confirmed_rr_path(sid)
    if not path_pref:
        print(f"[warn] 未在确认目录找到 {sid} 的 RR 文件，跳过。")
        return

    df = None
    if ".csv" in path_pref:
        df = _read_rr(path_pref[".csv"])
    elif ".parquet" in path_pref:
        df = _read_rr(path_pref[".parquet"])
    if df is None or df.empty or not {"t_s", "rr_ms"}.issubset(df.columns):
        print(f"[warn] {sid} RR 文件为空/缺列，跳过。")
        return

    rr = df["rr_ms"].to_numpy(dtype=float)
    ts = df["t_s"].to_numpy(dtype=float)
    n  = len(rr)

    # 前后两秒 rr 速度差别，调大＝更宽容，调小＝更严格。数据很毛躁就放宽一点
    rel_thr = RR_REL_THR
    loc_thr = RR_REL_THR
    # 标注最低和最高的rr,剔除超过或低于这些极端值的rr
    RR_MIN, RR_MAX = RR_LOW_MS, RR_HIGH_MS
    # 修复长度，值为rr点数，太小会修不完，太大容易把正常起伏也拉平
    MAX_SEG_BEATS  = RUN_MAX   
    
    # 识别突然下降的rr,0.85 意味着低于 85% 基线就算突然下降
    LOW_RUN_RATIO = 0.9   # 识别 RR 值低于局部中位数的比例，越低越宽松
    # 要连续多少个低点才承认是“突然下降”，数据越碎，用低值
    LOW_RUN_MINLEN = 1      # 阶梯串最小长度
    # 不均匀的长短拍。这两个数是“短得多短才算短”“长得多长才算长”的尺子
    # 调小这两个＝更容易合并（激进），调大＝更保守
    SLC_SHORT = 0.7         # 例如短于基线的 65% 算“短”
    SLC_LONG  = 1.3           # 例如长于基线的 140% 算“长”
    # 改置提高能提升将两个短拍合成一个拍的概率
    # 把“短+长两拍的总路程”和“正常两拍总路程”比一比
    # 相差不超过这个比例就认定“确实被拆了”，就合并回去
    # 0.30 表示差 30% 以内都算能接受。
    # 调大＝更容易合并（但可能误并），调小＝更谨慎。该值已经在全局变量中设置，这里不再重复设置
    # COMP_TOL = 0.3
    

    # 供后续规则使用的初始化
    med = np.full(n, np.nan)
    extra_segs = []
    extra_bad = np.zeros(n, dtype=bool)

    # -------- 1) 标记 --------
    # bad_len 合法 rr 所在的 绝对阈值区间
    bad_len = (rr < RR_MIN) | (rr > RR_MAX)
    # bad_rel 合法 rr 相对变化
    bad_rel = np.zeros(n, dtype=bool)
    if n >= 2:
        prev = rr[:-1]
        drel = np.zeros(n); drel[1:] = np.abs(np.diff(rr)) / np.maximum(prev, 1e-6)
        bad_rel = drel > rel_thr

    # bad_loc 局部中位数，5 拍
    bad_loc = np.zeros(n, dtype=bool)
    if n >= 5:
        pad = 2
        rr_pad = np.r_[np.repeat(rr[0], pad), rr, np.repeat(rr[-1], pad)]
        med = np.empty(n)
        for i in range(n):
            win = rr_pad[i:i+2*pad+1]
            m = np.nanmedian(win)
            med[i] = m if np.isfinite(m) else rr[i]
        bad_loc = np.abs(rr - med) / np.maximum(med, 1e-6) > loc_thr

        if n >= 5:
            low_band = LOW_RUN_RATIO * med
            low_mask = rr < np.maximum(low_band, 1e-6)

            # 把连续 True 聚合为段
            i = 0
            while i < n:
                if not low_mask[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n and low_mask[j + 1]:
                    j += 1
                run_len = j - i + 1
                if run_len >= LOW_RUN_MINLEN:
                    # 这是一段“拆拍式”的低 RR 串，交给后续 pair-merge + 插值
                    extra_segs.append((i, j))
                    extra_bad[i:j + 1] = True  # 先记录为额外坏点，稍后合并到 bad
                i = j + 1

    bad = bad_len | bad_rel | bad_loc
    bad |= extra_bad  # 合并“阶梯串识别”产生的坏点

    # -------- 2) 段聚合（允许端点），并做“成对补偿合并” --------
    segs = []
    i = 0
    while i < n:
        if not bad[i]:
            i += 1; continue
        j = i
        while j + 1 < n and bad[j + 1]:
            j += 1
        seg_len = j - i + 1
        if 1 <= seg_len <= MAX_SEG_BEATS:
            segs.append((i, j))
        i = j + 1

    if extra_segs:
        segs.extend(extra_segs)
        # 合并重叠/相邻段，避免重复处理
        segs.sort()
        merged = []
        cur_s, cur_e = segs[0]
        for s, e in segs[1:]:
            if s <= cur_e + 1:     # 相邻也合并
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        segs = merged
        
    if not segs:
        _write_rr(path_pref, df)
        summary_records.append({"subject_id": sid, "n_segments": 0, "n_corrected": 0,
                                "ratio_corrected": 0.0, "warning": ""})
        return

    rr_fix = rr.copy()
    corrected = 0

    # 成对补偿合并：两拍短 RR 的和≈局部 2×中位数
    def _pair_merge(i0, i1):
        nonlocal rr_fix, corrected
        k = i0
        while k + 1 <= i1:
            # 局部中位数用邻域 5 拍估计
            lo = max(0, k - 2)
            hi = min(n, k + 3)
            mloc = np.nanmedian(rr_fix[lo:hi]) if hi - lo >= 3 else np.nanmedian(rr_fix[max(0,k-1):min(n,k+2)])
            if not np.isfinite(mloc) or mloc <= 0:
                mloc = rr_fix[k]
            if abs((rr_fix[k] + rr_fix[k+1]) - 2.0*mloc) / (2.0*mloc) < COMP_TOL:
                rr_fix[k+1] = rr_fix[k] + rr_fix[k+1]  # 合并到后一拍
                rr_fix[k]   = np.nan                   # 前一拍置 NaN，后面统一插值
                corrected += 1
            k += 2

    # 形态：rr[k] < SLC_SHORT*med[k] 且 rr[k+1] > SLC_LONG*med[k+1]
    # 或反向（长-短），都视作漏/重检，执行“合并到后一拍”
    def _hard_slc_fix(i0, i1):
        nonlocal rr_fix, corrected
        k = i0
        while k + 1 <= i1:
            m0 = med[k]   if np.isfinite(med[k])   and med[k]   > 0 else rr_fix[k]
            m1 = med[k+1] if np.isfinite(med[k+1]) and med[k+1] > 0 else rr_fix[k+1]
            cond_short_long = (rr_fix[k]   < SLC_SHORT * m0) and (rr_fix[k+1] > SLC_LONG * m1)
            cond_long_short = (rr_fix[k]   > SLC_LONG  * m0) and (rr_fix[k+1] < SLC_SHORT * m1)
            if cond_short_long or cond_long_short:
                rr_fix[k+1] = rr_fix[k] + rr_fix[k+1]
                rr_fix[k]   = np.nan
                corrected  += 1
                k += 2
                continue
            k += 1

    # 先做硬补偿，再做你的成对补偿
    for (a, b) in segs:
        if b - a + 1 >= 2:
            _hard_slc_fix(a, b)
            _pair_merge(a, b)

    # -------- 3) 插值修复（允许端点）--------
    # 先把仍标记为 bad 的位置置 NaN
    nan_mask = bad | np.isnan(rr_fix)
    rr_fix[nan_mask] = np.nan

    # 用“时间”为自变量做插值；端点用最近邻延拓
    good = np.isfinite(rr_fix)
    if good.sum() >= 2:
        rr_interp = rr_fix.copy()
        # 内部线性
        rr_interp[~good] = np.interp(ts[~good], ts[good], rr_fix[good])
        # 左端/右端外推：最近邻
        if np.isnan(rr_interp[0]):
            rr_interp[0] = rr_interp[np.flatnonzero(~np.isnan(rr_interp))[0]]
        if np.isnan(rr_interp[-1]):
            rr_interp[-1] = rr_interp[np.flatnonzero(~np.isnan(rr_interp))[-1]]
        corrected += int(np.sum(nan_mask))
        rr_fix = rr_interp

    # -------- 4) 轻度平滑（仅在修复过的邻域）--------
    if corrected > 0 and n >= 3:
        mark = np.zeros(n, dtype=bool)
        for a, b in segs:
            mark[max(0, a-1):min(n, b+2)] = True
        rr_sm = rr_fix.copy()
        for k in range(1, n-1):
            if mark[k]:
                rr_sm[k] = np.median([rr_fix[k-1], rr_fix[k], rr_fix[k+1]])
        rr_fix = rr_sm

    # -------- 5) 可选：hrvanalysis 兜底 --------
    if remove_outliers and remove_ectopic_beats and interpolate_nan_values:
        rri = rr_fix.copy()
        # 越界 → NaN
        rri[(rri < RR_MIN) | (rri > RR_MAX)] = np.nan
        # hrvanalysis pipeline：离群→异位搏→插值。method 兼容库支持的 4 种口径
        _ectopic_method = str(PARAMS.get("hrv_ectopic_method", "malik")).lower()
        _allowed_methods = {"malik", "kamath", "karlsson", "acar"}
        if _ectopic_method not in _allowed_methods:
            _ectopic_method = "malik"
        rri2 = remove_outliers(rri, low_rri=RR_MIN, high_rri=RR_MAX, verbose=False)
        # 某些版本的 hrvanalysis 不接受 verbose 参数，这里显式不传
        try:
            rri3 = remove_ectopic_beats(rri2, method=_ectopic_method)
        except TypeError:
            rri3 = remove_ectopic_beats(rri2, method=_ectopic_method)
        rri4 = interpolate_nan_values(rri3, interpolation_method="linear")
        # 只在修复过的区域替换，避免整体口径漂移
        replace_mask = (np.isnan(rr) | bad).astype(bool)

        # hrvanalysis 返回的常是 list，先转成 ndarray 并做长度对齐
        rri4_arr = np.asarray(rri4, dtype=float)
        if rri4_arr.shape[0] != rr_fix.shape[0]:
            # 极少数情况下长度不一致：按较短长度对齐，剩余位置保留原值
            m = min(rri4_arr.shape[0], rr_fix.shape[0])
            tmp = rr_fix.copy()
            tmp[:m] = np.where(replace_mask[:m], rri4_arr[:m], rr_fix[:m])
            rr_fix = tmp
        else:
            rr_fix = np.where(replace_mask, rri4_arr, rr_fix)

    # -------- 6) 写回 & 汇总 --------
    df_out = df.copy()
    df_out["rr_ms"] = rr_fix
    _write_rr(path_pref, df_out)

    ratio_corr = (np.sum(bad) / max(1, n))
    summary_records.append({
        "subject_id": sid, "n_segments": len(segs),
        "n_corrected": int(np.sum(bad)),
        "ratio_corrected": round(float(ratio_corr), 4),
        "warning": ""
    })


def main():
    dec_path = SELECT_RR_REF_DIR / "decision.csv"
    if not dec_path.exists():
        print(f"[exit] 未找到 {dec_path}。请先完成上一阶段（生成 decision.csv）后再运行本脚本。")
        return
    dec = pd.read_csv(dec_path)
    cols = {c.lower(): c for c in dec.columns}
    sid_col = cols.get("subject_id", None)
    choice_col = cols.get("choice_suggested", None)
    if sid_col is None or choice_col is None:
        print("[exit] decision.csv 缺少必要列 subject_id / choice_suggested。")
        return
    rows = dec[[sid_col, choice_col]].rename(columns={sid_col: "subject_id", choice_col: "choice"})
    # 选择处理哪类 rr 数据。默认都处理 device_rr 或 ecg_rr 的被试
    tasks = rows[rows["choice"].astype(str).str.lower().isin(["device_rr","ecg_rr"])]
    if tasks.empty:
        print("[info] 无需处理：decision.csv 中没有选择 device_rr 或 ecg_rr 的被试。")
        return

    summary_records: List[Dict] = []
    for sid, choice in tasks[["subject_id", "choice"]].astype(str).itertuples(index=False):
        ch = (choice or "").strip().lower()
        if ch == "device_rr":
            clean_device_rr(sid, summary_records)
        elif ch == "ecg_rr":
            clean_ecgv2_rr(sid, summary_records)
        else:
            print(f"[skip] {sid}: 不应该出现的 choice={choice}, 跳过")

    # 输出汇总 QC
    if summary_records:
        qc = pd.DataFrame(summary_records)
        qc_path = OUT_ROOT / "clean_rr_summary.csv"
        qc.to_csv(qc_path, index=False)
        print(f"[done] 清理完成：{len(summary_records)} 个被试；QC 汇总已写入 {qc_path}")
    else:
        print("[info] 未产生任何输出。")

if __name__ == "__main__":
    main()