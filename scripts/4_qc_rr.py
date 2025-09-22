# scripts/2_qc_rr.py
# 目的：对“原始 ECG”做快速质量检查（Quality Check）。
# 输出：
#   1) 快速 RR 标注表：data/processed/2qc_rr_<sid>.csv（含 quality_check 布尔列）
#   2) QC 图：data/processed/qc/<sid>_qc.png（蓝=HR，橙=可疑RR，红=建议排查区间）
#
# 读取的数据源：
#   - 来自 settings.RAW_CACHE_DIR（你可在 settings.py 里修改这个路径）
#   - 文件名规范：<subject_id>_ecg.parquet 或 <subject_id>_ecg.csv
#   - parquet/csv 期望的列：
#       必需：time_s, value
#       可选：fs_hz, subject_id, signal
#     若没有 fs_hz，将从 time_s 差分估算采样率（取中位数）
#
# 可调规则（在 settings.PARAMS 里改，不要改本脚本）：
#   rr_artifact_threshold  # 相邻 RR 变动阈值（默认 0.20）
#   hr_min_max_bpm         # 心率越界范围，如 (35, 140) 或活动期放宽到 (40, 180)
#   rr_fix_strategy        # 此脚本只做“标注”，不做修复；该字段仅回显
#
# 使用：
#   在 VSCode 直接运行本脚本。控制台会打印：
#     - 使用的原始数据路径
#     - 规则口径
#     - 每个被试的可疑比例与产物路径
#
# 说明：
#   这是“现场/采集后立刻”的快速体检工具。它不删除数据、不生成 cutlist 文件。
#   半透明红条是基于滑窗质量的“建议排查区间”，用于肉眼复核。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- project-root bootstrap ---
import sys
from pathlib import Path
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent))
        break
    _p = _p.parent
# --- end bootstrap ---

# 读取 settings
from settings import PROCESSED_DIR, RAW_CACHE_DIR, PARAMS
from settings import DATASETS, ACTIVE_DATA, DATA_DIR, PROCESSED_DIR, PARAMS

PRE_SID = DATASETS[ACTIVE_DATA]["preview_sids"]

# 可选进度条
try:
    from tqdm import tqdm
    def track(it, total, desc): return tqdm(it, total=total, desc=desc)
except Exception:
    def track(it, total, desc):
        print(f"[progress] {desc} total={total}")
        for i, x in enumerate(it, 1):
            if i == 1 or i == total or i % max(1, total//10) == 0:
                print(f"[progress] {desc} {i}/{total}")
            yield x

# 质检滑窗参数（只影响红条绘图，非硬删除）
QC_WIN_S    = 30.0   # 滑窗长度（秒）
QC_STRIDE_S = 10.0   # 步长（秒）
QC_MIN_NRR  = 20     # 窗内最少 RR 数
QC_MIN_VALID= 0.85   # 窗内 valid 比例阈值
QC_MAX_FLAG = 0.20   # 窗内 flagged 比例阈值
QC_HR_JUMP  = 25.0   # 相邻窗 HR 均值跳变阈值（bpm）
QC_MERGE_GAP= 5.0    # 合并小间隙（秒）
QC_MIN_CUT  = 3.0    # 最短建议片段（秒）

# 可选：只跑部分被试，例如 ["f1y01","f1o01"], [] 则检查全部被试数据
SUBJECTS_FILTER: list[str] = ["f1y01","f1o01"]  #

# 输入输出目录
paths = DATASETS[ACTIVE_DATA]["paths"]
SRC_NORM_DIR = (DATA_DIR / paths["confirmed"]).resolve()
CLEAN_OUT_DIR   = (DATA_DIR / paths["clean"]).resolve()
CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)

def _find_raw_files() -> list[Path]:
    # 支持两种扩展名，递归搜索
    exts = ["*_ecg.parquet", "*_ecg.csv"]
    files = []
    for pat in exts:
        files += list(RAW_CACHE_DIR.rglob(pat))
    files = sorted(files)
    if SUBJECTS_FILTER:
        files = [f for f in files if f.stem.replace("_ecg","") in SUBJECTS_FILTER]
    return files

def _read_ecg(path: Path) -> tuple[np.ndarray, np.ndarray, float, str]:
    """返回 time_s, ecg, fs_hz, subject_id"""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    cols = set(df.columns)
    if not {"time_s","value"}.issubset(cols):
        raise ValueError(f"{path.name} 缺少列：需要 ['time_s','value']，可选 ['fs_hz','subject_id','signal']")
    time_s = df["time_s"].to_numpy(dtype=float)
    ecg = df["value"].to_numpy(dtype=float)
    # 采样率优先用列，否则估算
    if "fs_hz" in cols:
        fs = float(df["fs_hz"].iloc[0])
    else:
        dt = np.median(np.diff(time_s))
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError(f"{path.name} 时间轴无法估算采样率")
        fs = float(round(1.0 / dt))
    # 被试号
    if "subject_id" in cols:
        sid = str(df["subject_id"].iloc[0])
    else:
        sid = path.stem.replace("_ecg","")
    return time_s, ecg, fs, sid

def _quick_rpeaks(time_s: np.ndarray, ecg: np.ndarray, fs: float) -> np.ndarray:
    # 用轻量方案：neurokit2 的 ecg_process，返回 R 峰索引
    import neurokit2 as nk
    _, info = nk.ecg_process(ecg, sampling_rate=fs)
    rpeaks = info.get("ECG_R_Peaks", None)
    if rpeaks is None or len(rpeaks) < 3:
        return np.array([], dtype=int)
    return np.asarray(rpeaks, dtype=int)

def _mark_rr(time_s: np.ndarray, rpeaks: np.ndarray, fs: float,
             delta_thr: float, hr_min: float, hr_max: float) -> pd.DataFrame:
    rr_ms = np.diff(rpeaks) / fs * 1000.0
    t_rr  = time_s[rpeaks[1:]]
    hr    = 60000.0 / rr_ms
    # 规则标注
    rr_series = pd.Series(rr_ms)
    flag_delta = rr_series.pct_change().abs() > float(delta_thr)
    flag_delta.iloc[0] = False
    flag_hr = (hr < float(hr_min)) | (hr > float(hr_max))
    flagged = (flag_delta.to_numpy() | flag_hr)
    quality_check = flagged.copy()  # 就按你的要求：一个布尔列用于快速识别
    df = pd.DataFrame({
        "t_s": t_rr.astype(float),
        "rr_ms": rr_ms.astype(float),
        "hr_bpm": hr.astype(float),
        "valid": ~flagged,                 # 保留，便于后续观察
        "flagged": flagged,                # 保留，便于后续观察
        "flag_delta": flag_delta.to_numpy(),
        "flag_hr": flag_hr.astype(bool),
        "flag_reason": np.where(flagged,
                         np.where(flag_delta & flag_hr, "delta|hr",
                         np.where(flag_delta, "delta", "hr")),""),
        "quality_check": quality_check,    # 你要的总标记
        "n_rr_corrections": np.cumsum(flagged).astype(int)
    })
    return df

def _rolling_bad_spans(rr_df: pd.DataFrame) -> list[tuple[float,float]]:
    """基于滑窗的质量评估，只用于绘图红条提示，不导出文件。"""
    if rr_df.empty:
        return []
    t0, t1 = float(rr_df["t_s"].min()), float(rr_df["t_s"].max())
    starts = np.arange(t0, max(t0, t1-QC_WIN_S)+1e-9, QC_STRIDE_S)
    rolls = []
    for s in starts:
        e = s + QC_WIN_S
        sub = rr_df[(rr_df["t_s"]>=s) & (rr_df["t_s"]<e)]
        n = len(sub)
        if n < QC_MIN_NRR:
            bad = True; reason = "too_few_rr"
            vr = 0.0; fr = 1.0; hrm = np.nan
        else:
            vr = float(sub["valid"].mean())
            fr = float(sub["flagged"].mean())
            hrm= float(sub["hr_bpm"].mean())
            bad = (vr < QC_MIN_VALID) or (fr > QC_MAX_FLAG)
            reason = "low_valid" if vr < QC_MIN_VALID else ("high_flag" if fr > QC_MAX_FLAG else "")
        rolls.append({"s":s,"e":e,"bad":bad,"hr_mean":hrm,"reason":reason})
    # HR 跳变
    for i in range(1, len(rolls)):
        h0, h1 = rolls[i-1]["hr_mean"], rolls[i]["hr_mean"]
        if np.isfinite(h0) and np.isfinite(h1) and abs(h1-h0) > QC_HR_JUMP:
            rolls[i]["bad"] = True
            rolls[i]["reason"] = (rolls[i]["reason"] + "|hr_jump").strip("|")
    # 合并坏窗
    bads = [r for r in rolls if r["bad"]]
    if not bads:
        return []
    bads.sort(key=lambda x: x["s"])
    merged = []
    cur = {"s":bads[0]["s"], "e":bads[0]["e"]}
    for r in bads[1:]:
        if r["s"] <= cur["e"] + QC_MERGE_GAP:
            cur["e"] = max(cur["e"], r["e"])
        else:
            merged.append(cur); cur = {"s":r["s"], "e":r["e"]}
    merged.append(cur)
    # 最短过滤
    merged = [m for m in merged if (m["e"]-m["s"]) >= QC_MIN_CUT]
    return [(round(m["s"],3), round(m["e"],3)) for m in merged]

def _plot_qc(sid: str, rr_df: pd.DataFrame, spans: list[tuple[float,float]], out_png: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    if not rr_df.empty:
        ax.plot(rr_df["t_s"], rr_df["hr_bpm"], linewidth=0.8, label="HR (bpm)")
        bad = rr_df[rr_df["quality_check"]]
        if len(bad):
            ax.scatter(bad["t_s"], bad["hr_bpm"], s=8, alpha=0.7, label="suspect RR")
        for s, e in spans:
            ax.axvspan(s, e, color="red", alpha=0.25)
    ax.set_title(f"{sid} — QC overview")
    ax.set_xlabel("time (s)"); ax.set_ylabel("HR (bpm)")
    ax.legend(loc="upper right"); ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def main():
    print("[qc] 读取原始 ECG 的根路径（可在 settings.py 修改 RAW_CACHE_DIR） →", RAW_CACHE_DIR)
    print("[qc] 规则口径：delta_thr=%.2f  hr_range=%s  strategy=%s" %
          (PARAMS.get("rr_artifact_threshold", 0.20),
           str(PARAMS.get("hr_min_max_bpm", (35,140))),
           PARAMS.get("rr_fix_strategy","delete")))

    raw_files = _find_raw_files()
    if not raw_files:
        print("[error] 未找到原始 ECG。请确认：")
        print("  1) settings.RAW_CACHE_DIR 指向你的原始数据根目录；")
        print("  2) 文件名形如 <subject_id>_ecg.parquet 或 <subject_id>_ecg.csv；")
        print("  3) 表头至少包含 ['time_s','value']，可选 ['fs_hz','subject_id','signal']")
        sample = RAW_CACHE_DIR / "your_subject_ecg.parquet"
        print("  样例路径：", sample)
        return

    out_rows = []
    for path in track(raw_files, total=len(raw_files), desc="QC"):
        try:
            time_s, ecg, fs, sid = _read_ecg(path)
        except Exception as e:
            print(f"[skip] {path.name} 读取失败：{e}")
            continue

        print(f"[qc] {sid}: 原始文件 → {path}")

        # R 峰 + RR 标注
        rpeaks = _quick_rpeaks(time_s, ecg, fs)
        if len(rpeaks) < 3:
            print(f"[warn] {sid}: R 峰不足，跳过。")
            continue
        thr = float(PARAMS.get("rr_artifact_threshold", 0.20))
        hr_min, hr_max = PARAMS.get("hr_min_max_bpm", (35,140))
        rr_df = _mark_rr(time_s, rpeaks, fs, thr, hr_min, hr_max)

        # 图片中质检红条，内部为可疑数据
        spans = _rolling_bad_spans(rr_df)

        # 合并到 quality_check：点级命中 或 落在红条区间 都算可疑
        if spans:
            in_span = np.zeros(len(rr_df), dtype=bool)
            for s, e in spans:
                in_span |= ((rr_df["t_s"] >= s) & (rr_df["t_s"] < e)).to_numpy()
            rr_df["quality_check"] = rr_df["flagged"] | in_span
        else:
            rr_df["quality_check"] = rr_df["flagged"]

        # 保存快速 RR
        out_csv = PROCESSED_DIR / f"2qc_rr_{sid}.csv"
        rr_df.to_csv(out_csv, index=False)

        # 保存图
        out_png = CLEAN_OUT_DIR / f"{sid}_qc.png"
        _plot_qc(sid, rr_df, spans, out_png)

        # 简短统计
        pct_bad = 100.0 * float(rr_df["quality_check"].mean()) if len(rr_df) else 0.0
        print(f"[ok] {sid}: 2qc_rr → {out_csv.name}  | suspect={pct_bad:.2f}%  | 图 → {out_png.name}")

        out_rows.append({"subject_id": sid,
                         "n_rr": int(len(rr_df)),
                         "pct_suspect": round(pct_bad, 2)})

    if out_rows:
        summ = pd.DataFrame(out_rows)
        summ_path = PROCESSED_DIR / "2qc_summary.csv"
        summ.to_csv(summ_path, index=False)
        print(f"[save] 概览 → {summ_path}")

    print("\n[hint] 若你处于非静息任务，建议在 settings.PARAMS 中放宽阈值，如：")
    print("       rr_artifact_threshold=0.25；hr_min_max_bpm=(40,160) 或 (40,180)")
    print("       仅需改 settings.py，不用改脚本。")

if __name__ == "__main__":
    main()