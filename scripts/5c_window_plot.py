from __future__ import annotations
import os, sys, re, json
from pathlib import Path
import pandas as pd
import numpy as np

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent))
        break
    _p = _p.parent
# --------------------------------------------------------
from settings import DATASETS, ACTIVE_DATA, DATA_DIR, SCHEMA, PARAMS

# 窗口数据载入路径，以及绘图的保存路径
DS = DATASETS[ACTIVE_DATA]
paths = DS["paths"]
SRC_ROOT      = (DATA_DIR / paths["windowing"]).resolve()
SRC_DIR_RR       = (SRC_ROOT / "collected"/ "rr").resolve()
SRC_DIR_ECG       = (SRC_ROOT / "collected"/ "ecg").resolve()
OUT_ROOT      = (DATA_DIR / paths["preview"]).resolve()
OUT_DIR       = (OUT_ROOT / 'window_plots').resolve()

# ----- 注意 -------
# RR 文件以 被试ID_rr_窗号命名，例如 P003S001T001R001_rr_w01.csv
# rr数据的前两列固定为 't_s', 'rr_ms'
# 生成的结果包括：
# 1. 多个被试的hr整合图：在同一窗口下展示多个被试的hr曲线图，不同被试ID用不同颜色表示。使用固定的 Y 轴范围（e.g., plt.ylim(50, 120)）。多个hr组合时，t_s 时间轴要能包含所有被试hr的时间范围。生成的图命名为 {win_id}_hr_comparison.png
# 2. RR 原始信号图：每个被试在同一窗口上绘制独立的rr原始信号图，原始RR图须展示被试窗口内RR数据波动全貌，Y 轴范围根据数据调整。时间轴 t_s 做去基线处理 df['t_s_relative'] = df['t_s'] - df['t_s'].min()，然后使用这个 t_s_relative 来绘图，这样所有被试的曲线都会从 0 秒开始，更便于比较。生成的图命名为 {win_id}_{sub_id})_rr.png

# 要绘制的被试ID列表与窗口ID列表，按照这个列表生成预览图
SID = ["P001S001T001R001","P002S001T002R001","P003S001T001R001","P004S001T002R001"]
WIN = ['w02','w06']

import matplotlib.pyplot as plt

# -------------------- configurable plotting params --------------------
HR_YLIM = (50, 120)  # fixed y-limits for HR comparison plots
HR_LABEL = "HR (bpm)"
RR_LABEL = "RR interval (ms)"
T_LABEL_ABS = "Time (s)"
T_LABEL_REL = "Time since window start (s)"

# ensure output dir exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _rr_path_for(sid: str, win_id: str) -> Path:
    """
    Build path to RR csv for a given subject and window.
    Expected filename pattern: {sid}_rr_{win_id}.csv
    """
    fname = f"{sid}_rr_{win_id}.csv"
    return (SRC_DIR_RR / fname).resolve()

def _load_rr_df(path: Path) -> pd.DataFrame | None:
    """
    Load RR csv with at least two columns: 't_s' and 'rr_ms'.
    Returns a DataFrame with added columns:
      - 't_s_relative' : t_s zeroed to its own minimum
      - 'hr_bpm'       : 60000 / rr_ms
    """
    if not path.exists():
        print(f"[warn] missing RR file: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}")
        return None
    required = {"t_s", "rr_ms"}
    if not required.issubset(set(df.columns)):
        print(f"[warn] file missing required columns {required}: {path}")
        return None
    # drop obvious NaNs
    df = df.dropna(subset=["t_s", "rr_ms"]).copy()
    # ensure numeric
    df["t_s"] = pd.to_numeric(df["t_s"], errors="coerce")
    df["rr_ms"] = pd.to_numeric(df["rr_ms"], errors="coerce")
    df = df.dropna(subset=["t_s", "rr_ms"])
    if df.empty:
        print(f"[warn] empty after cleaning: {path}")
        return None
    # derived
    df["t_s_relative"] = df["t_s"] - df["t_s"].min()
    # protect against zero/negative rr
    df = df[df["rr_ms"] > 0].copy()
    df["hr_bpm"] = 60000.0 / df["rr_ms"]
    return df

def plot_hr_comparison_for_window(win_id: str, sids: list[str]) -> None:
    """
    Plot HR curves for multiple subjects on the same axes
    using absolute time (t_s). The x-axis spans the min/max
    over all available subjects in this window.
    Save figure as {win_id}__hr_comparison.png
    """
    series = []
    for sid in sids:
        path = _rr_path_for(sid, win_id)
        df = _load_rr_df(path)
        if df is None or df.empty:
            continue
        series.append((sid, df))
    if not series:
        print(f"[info] no data to plot HR comparison for window {win_id}")
        return
    # build plot (aligned to window start per subject)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xmin = 0.0
    xmax = max(df["t_s_relative"].max() for _, df in series)
    for sid, df in series:
        ax.plot(df["t_s_relative"], df["hr_bpm"], label=sid, linewidth=1.2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(*HR_YLIM)
    ax.set_xlabel(T_LABEL_REL)
    ax.set_ylabel(HR_LABEL)
    ax.set_title(f"HR comparison (aligned to window start) | window {win_id}")
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    out_path = OUT_DIR / f"{win_id}_hr_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[ok] saved {out_path}")

def plot_rr_per_subject_for_window(win_id: str, sids: list[str]) -> None:
    """
    For each subject in this window, plot RR vs t_s_relative
    as an individual figure using scatter points (tachogram "band").
    Save as {win_id}_{sid}_rr.png
    """
    for sid in sids:
        path = _rr_path_for(sid, win_id)
        df = _load_rr_df(path)
        if df is None or df.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 3.8))
        # scatter-style tachogram: no lines, small markers, slight transparency
        ax.scatter(
            df["t_s_relative"], df["rr_ms"],
            s=8, alpha=0.8, linewidths=0, edgecolors="none"
        )
        ax.set_xlabel(T_LABEL_REL)
        ax.set_ylabel(RR_LABEL)
        ax.set_title(f"RR tachogram (scatter) | {sid} | window {win_id}")
        # auto y-limits based on data with 5% padding
        ymin, ymax = float(df["rr_ms"].min()), float(df["rr_ms"].max())
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 10.0
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        out_path = OUT_DIR / f"{win_id}_{sid}_rr.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"[ok] saved {out_path}")

def plot_ecg_per_subject_for_window(win_id: str, sid: list[str]) -> None:
    """
    在窗口级 ECG 文件({sid}_ecg_{win_id}.csv)上绘制两行图：
    上：ECG 原始/带通 + R peaks；下：RR(ms) 并标记可疑 RR（不删除）
    仅依赖本脚本已有的导入(pd/np/plt)，不动其他函数。
    """
    # 可选依赖：SciPy 用于带通与 find_peaks；没有也能跑
    try:
        from scipy.signal import butter, filtfilt, find_peaks  # type: ignore
        _have_scipy = True
    except Exception:
        _have_scipy = False

    for s in sid:
        ecg_fname = f"{s}_ecg_{win_id}.csv"
        path = (SRC_DIR_ECG / ecg_fname).resolve()
        if not path.exists():
            print(f"[warn] missing ECG file: {path}")
            continue

        # 直接用 pandas 读取 ECG
        try:
            df_raw = pd.read_csv(path)
        except Exception as e:
            print(f"[warn] failed to read ECG {path}: {e}")
            continue

        # 自适应列名
        time_cols = ["time_s", "t_s", "time", "timestamp_s"]
        val_cols  = ["value", "ecg", "ecg_mv", "signal"]
        col_time = next((c for c in time_cols if c in df_raw.columns), None)
        col_val  = next((c for c in val_cols  if c in df_raw.columns), None)
        if col_time is None or col_val is None:
            print(f"[warn] {s}: ECG missing time/value columns in {path.name}")
            continue

        df_raw = df_raw[[col_time, col_val]].rename(columns={col_time: "time_s", col_val: "value"})
        df_raw = df_raw.dropna(subset=["time_s", "value"]).sort_values("time_s")
        if df_raw.empty:
            print(f"[warn] {s}: empty ECG for window {win_id}")
            continue

        # 用文件自身时间范围作为窗口
        t0 = float(df_raw["time_s"].min())
        t1 = float(df_raw["time_s"].max())
        df = df_raw.copy()

        if df.shape[0] < 50:
            print(f"[info] {s}: too few ECG points in window {win_id}")
            continue

        t = df["time_s"].to_numpy(dtype=float)
        x = df["value"].to_numpy(dtype=float)

        # 采样率估计
        dt = np.median(np.diff(t)) if t.size >= 3 else np.nan
        if not np.isfinite(dt) or dt <= 0:
            print(f"[warn] {s}: invalid time base in {win_id}")
            continue
        fs = 1.0 / dt

        # 带通 5–20 Hz（有 SciPy 才做）
        if _have_scipy:
            low, high = 5.0, 20.0
            nyq = 0.5 * fs
            hi = min(high, 0.45 * fs)
            lo = min(low, hi * 0.9) if low >= hi else low
            try:
                b, a = butter(3, [lo/nyq, hi/nyq], btype="band")
                xf = filtfilt(b, a, x)
            except Exception:
                xf = x
        else:
            xf = x

        # 峰检测
        min_dist = int(max(1, round(0.30 * fs)))  # >= 0.3s
        q1, q99 = np.nanpercentile(xf, [1, 99])
        prom = max(1e-3, 0.2 * (q99 - q1))
        if _have_scipy:
            try:
                pk, _ = find_peaks(xf, distance=min_dist, prominence=prom)
            except Exception:
                pk = np.array([], dtype=int)
        else:
            # 简易兜底
            y = xf
            dy_prev = np.r_[np.nan, np.diff(y)]
            dy_next = np.r_[np.diff(y), np.nan]
            cand = np.where((dy_prev > 0) & (dy_next < 0))[0]
            cand = cand[np.argsort(y[cand])[::-1]]
            kept = []
            for i in cand:
                if any(abs(i - k) < min_dist for k in kept):
                    continue
                left = max(0, i - min_dist)
                right = min(len(y) - 1, i + min_dist)
                base = min(np.nanmin(y[left:i+1]) if i > left else y[i],
                           np.nanmin(y[i:right+1]) if right > i else y[i])
                if (y[i] - base) >= prom:
                    kept.append(i)
            kept.sort()
            pk = np.asarray(kept, dtype=int)

        # RR 与 QC 标记（不删除）
        rr_ms, qc_flags, tm = None, None, None
        if pk.size >= 2:
            rr_s = np.diff(t[pk])
            rr_ms = 1000.0 * rr_s
            rr = np.asarray(rr_ms, dtype=float)
            flags_len = (rr < 300.0) | (rr > 2000.0)
            rel = np.zeros_like(rr, dtype=bool)
            if rr.size >= 2:
                prev = rr[:-1]
                d = np.abs(np.diff(rr)) / np.maximum(prev, 1e-6)
                rel[1:] = d > 0.25
            qc_flags = flags_len | rel
            tm = (t[pk][1:] + t[pk][:-1]) / 2.0

            rr_med_all = float(np.nanmedian(rr)) if rr.size else np.nan
            hr_med_all = float(60000.0 / rr_med_all) if np.isfinite(rr_med_all) and rr_med_all > 0 else np.nan
            rmssd_all = float(np.sqrt(np.nanmean(np.diff(rr) ** 2))) if rr.size >= 2 else np.nan
            if np.any(~qc_flags) and np.sum(~qc_flags) >= 2:
                rr_clean = rr[~qc_flags]
                rr_med_clean = float(np.nanmedian(rr_clean))
                hr_med_clean = float(60000.0 / rr_med_clean) if rr_med_clean > 0 else np.nan
                rmssd_clean = float(np.sqrt(np.nanmean(np.diff(rr_clean) ** 2))) if rr_clean.size >= 2 else np.nan
            else:
                rr_med_clean = hr_med_clean = rmssd_clean = np.nan
            print(
                f"[stats] {s} {win_id} t=[{t0:.1f},{t1:.1f}]s | "
                f"RR_med(all)={rr_med_all:.1f} ms, HR_med(all)={hr_med_all:.1f} bpm, RMSSD(all)={rmssd_all:.1f} ms; "
                f"RR_med(clean)={rr_med_clean:.1f} ms, HR_med(clean)={hr_med_clean:.1f} bpm, RMSSD(clean)={rmssd_clean:.1f} ms"
            )
        
        rr_path = _rr_path_for(s, win_id)
        rr_df = _load_rr_df(rr_path)
        if rr_df is None or rr_df.empty:
            print(f"[warn] {s}: missing/empty RR file for window {win_id}, skip ORIG-RR plot")
            rr_df = None

        # 绘图（三行）：
        #  1) ECG 原始/带通 + R peaks
        #  2) ECG-RR 折线（从本窗口ECG检测得到的RR）
        #  3) ORIG-RR 折线（从窗口RR文件加载的原始RR）
        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
        )
        # 行1：ECG（时间改为相对窗口起点，保证与下两行共享 x 轴一致）
        t_rel = t - t0
        ax0.plot(t_rel, x, color="#a6bddb", linewidth=0.6, alpha=0.8, label="ECG raw")
        ax0.plot(t_rel, xf, color="#225ea8", linewidth=0.8, alpha=0.9, label="ECG (5–20 Hz)")
        if pk.size:
            ax0.scatter(t_rel[pk], xf[pk], color="crimson", s=18, zorder=5, label="R peaks")
        ax0.set_xlim(0, t1 - t0)
        ax0.set_ylabel("ECG (a.u.)")
        ax0.legend(loc="upper right", frameon=True)
        ax0.set_title(f"{s} | {win_id} | duration={t1 - t0:.1f}s | sfs≈{fs:.1f} Hz, peaks={pk.size}")
        # 行2：ECG-RR（以窗口起点对齐时间轴）
        if rr_ms is not None:
            tm_rel = tm - t0
            ax1.plot(tm_rel, rr_ms, color="#31a354", linewidth=1.2, label="ECG-RR")
            if qc_flags is not None and np.any(qc_flags):
                ax1.scatter(tm_rel[qc_flags], np.asarray(rr_ms)[qc_flags], marker='x', s=28,
                            linewidths=0.9, color='crimson', alpha=0.9, label='QC flagged')
            q1, q99 = np.nanpercentile(rr_ms, [5, 95])
            span = max(1, q99 - q1)
            ax1.set_ylim(q1 - 0.2*span, q99 + 0.2*span)
        else:
            ax1.text(0.5, 0.5, "too few peaks", transform=ax1.transAxes,
                     ha="center", va="center", color="gray")
            ax1.set_ylim(0, 1)
            ax1.set_yticks([])
        ax1.set_ylabel("ECG-RR (ms)")
        ax1.grid(True, alpha=0.2)
        # 行3：ORIG-RR（原始RR数据文件）
        if rr_df is not None and not rr_df.empty:
            ax2.plot(rr_df["t_s_relative"], rr_df["rr_ms"], color="#4E79A7", linewidth=1.0, label="ORIG-RR")
            q1o, q99o = float(rr_df["rr_ms"].quantile(0.05)), float(rr_df["rr_ms"].quantile(0.95))
            span_o = max(1, q99o - q1o)
            ax2.set_ylim(q1o - 0.2*span_o, q99o + 0.2*span_o)
            ax2.legend(loc="upper right", frameon=True)
        else:
            ax2.text(0.5, 0.5, "no ORIG-RR", transform=ax2.transAxes,
                     ha="center", va="center", color="gray")
            ax2.set_ylim(0, 1)
            ax2.set_yticks([])
        ax2.set_xlim(0, t1 - t0)
        ax2.set_xlabel("time since window start (s)")
        ax2.set_ylabel("ORIG-RR (ms)")
        ax2.grid(True, alpha=0.2)
        # 保存
        out = (OUT_DIR / f"{win_id}_{s}_ecg_zoom.png").resolve()
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"[ok] ecg-zoom → {out}")

def main() -> None:
    # loop over requested windows
    for win_id in WIN:
        print(f"\n[window] {win_id}")
        plot_hr_comparison_for_window(win_id, SID)
        plot_rr_per_subject_for_window(win_id, SID)
        plot_ecg_per_subject_for_window(win_id, SID)

if __name__ == "__main__":
    main()