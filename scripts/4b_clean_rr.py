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
try:
    from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values  # noqa: F401
except Exception:
    # 若未安装，也不阻断；核心逻辑不依赖这些函数
    remove_outliers = None  # type: ignore
    remove_ectopic_beats = None  # type: ignore
    interpolate_nan_values = None  # type: ignore

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

# —— 读配置（含新增 RR 清理参数，设置默认以避免缺失键导致崩溃） ——
RR_REL_THR = float(PARAMS.get("rr_artifact_threshold", 0.20))
HR_MIN, HR_MAX = PARAMS.get("hr_min_max_bpm", (35, 140))
RR_LOW_MS = 300.0
RR_HIGH_MS = 2000.0
RR_DABS_MS = float(PARAMS.get("rr_delta_abs_ms", 150.0))
RUN_MAX = int(PARAMS.get("rr_short_run_max_beats", 5))
COMP_TOL = float(PARAMS.get("rr_pair_comp_tolerance", 0.15))
GROUPS_WARN = int(PARAMS.get("rr_groups_warn", 4))
RATIO_WARN = float(PARAMS.get("rr_max_correct_ratio_warn", 0.05))
BOTH_BAD_THR = float(PARAMS.get("both_bad_rel_dev_thr", 0.15))
INTERP_METHOD = str(PARAMS.get("interp_method", "pchip")).lower()
MERGE_ENABLE = bool(PARAMS.get("pair_merge_enable", True))

RR_T = SCHEMA["rr"]["t"]
RR_V = SCHEMA["rr"]["v"]
ECG_T = SCHEMA["ecg"]["t"]
ECG_V = SCHEMA["ecg"]["v"]

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

def _apply_repairs(df: pd.DataFrame, segs: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
    """按标签对短段执行修复：普通→插值；错分→成对合并（可开关）。返回修复后的 DF 与每段的处理记录。"""
    rr = pd.to_numeric(df[RR_V], errors="coerce").to_numpy()
    tt = pd.to_numeric(df[RR_T], errors="coerce").to_numpy()
    n = rr.size
    to_nan = np.zeros(n, dtype=bool)
    to_drop = np.zeros(n, dtype=bool)
    records: List[Dict] = []
    for s in segs:
        i0, i1 = s["i_start"], s["i_end"]
        nlen = i1 - i0 + 1
        if s["compensation"] and MERGE_ENABLE and nlen >= 2:
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
                "t_start": float(tt[i0]), "t_end": float(tt[i1]), "n_beats": nlen,
                "method": "merge_pair+interp_last" if ((i1 - i0 + 1) % 2 == 1) else "merge_pair",
                "reason": "compensation"
            })
        else:
            to_nan[i0:i1 + 1] = True
            records.append({
                "t_start": float(tt[i0]), "t_end": float(tt[i1]), "n_beats": nlen,
                "method": "interp", "reason": "outlier"
            })
    # 删除合并掉的点
    keep = ~to_drop
    rr2 = rr[keep]
    tt2 = tt[keep]
    # 对需插值位置置 NaN（注意索引需要映射到删除后的新数组）
    # 构建原->新索引映射
    idx_map = np.full(n, -1, dtype=int)
    idx_map[keep] = np.arange(keep.sum(), dtype=int)
    to_nan_new = np.zeros(rr2.size, dtype=bool)
    src_nan_idx = np.where(to_nan)[0]
    for k in src_nan_idx:
        j = idx_map[k]
        if j >= 0:
            to_nan_new[j] = True
    rr2[to_nan_new] = np.nan
    # 插值
    rr_filled = _interp_local(tt2, rr2, to_nan_new)
    out = pd.DataFrame({RR_T: tt2, RR_V: rr_filled})
    return out, records

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
    # 仅处理被选择为 device_rr 的被试
    tasks = rows[rows["choice"].astype(str).str.lower().eq("device_rr")]
    if tasks.empty:
        print("[info] 无需处理：decision.csv 中没有选择 device_rr 的被试。")
        return

    summary_records: List[Dict] = []
    for sid in tasks["subject_id"].astype(str).tolist():
        path_pref = _load_confirmed_rr_path(sid)
        if not path_pref:
            print(f"[warn] 未在确认目录找到 {sid} 的 RR 文件，跳过。")
            continue
        df = None
        # 优先读 csv
        if ".csv" in path_pref:
            df = _read_rr(path_pref[".csv"]) 
        elif ".parquet" in path_pref:
            df = _read_rr(path_pref[".parquet"]) 
        if df is None or df.empty:
            print(f"[warn] {sid} RR 文件为空或不可读，跳过。")
            continue

        # 2) 识别短时尖峰（1..RUN_MAX）
        segs = _detect_short_spike_segments(df)
        if not segs:
            # 无短段，直接覆盖写回原文件（等价原样保持），并记录 0 修正
            _write_rr(path_pref, df)
            summary_records.append({"subject_id": sid, "n_segments": 0, "n_corrected": 0, "ratio_corrected": 0.0, "warning": ""})
            continue

        # 3) 提醒条件（不中断）
        n_corr_beats = int(sum(s["n_beats"] for s in segs))
        ratio_corr = n_corr_beats / max(1, len(df))
        warn_msgs = []
        if len(segs) >= GROUPS_WARN:
            warn_msgs.append(f"异常短段数量≥{GROUPS_WARN}")
        if ratio_corr >= RATIO_WARN:
            warn_msgs.append(f"需修正搏点占比≥{RATIO_WARN:.2f}")

        # 4–5) 参考 ecg_rr，若“两路皆异常”则仅写 QC，不做修复
        ecg_rr = _load_ecg_rr_from_norm(sid)
        both_bad = _both_bad(ecg_rr, segs) if ecg_rr is not None else False
        if both_bad:
            summary_records.append({
                "subject_id": sid, "n_segments": len(segs), "n_corrected": 0,
                "ratio_corrected": 0.0, "warning": "; ".join(["两路皆异常"] + warn_msgs)
            })
            # 不修改原文件
            continue

        # 6) 执行修复
        df_fixed, recs = _apply_repairs(df, segs)
        # 覆盖写回（与确认目录一致）
        _write_rr(path_pref, df_fixed)
        summary_records.append({
            "subject_id": sid, "n_segments": len(segs), "n_corrected": n_corr_beats,
            "ratio_corrected": round(ratio_corr, 4), "warning": "; ".join(warn_msgs)
        })

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