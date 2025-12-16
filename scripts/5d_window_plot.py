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
SRC_DIR_RESP       = (SRC_ROOT / "collected"/ "resp").resolve()
OUT_ROOT      = (DATA_DIR / paths["preview"]).resolve()
OUT_DIR       = (OUT_ROOT / 'window_plots').resolve()

# ----- 注意 -------
# RR 文件以 被试ID_rr_窗号命名，例如 P003S001T001R001_rr_w01.csv
# rr数据的前两列固定为 't_s', 'rr_ms'
# 生成的结果包括：
# 1. 多个被试的hr整合图：在同一窗口下展示多个被试的hr曲线图，不同被试ID用不同颜色表示。使用固定的 Y 轴范围（e.g., plt.ylim(50, 120)）。多个hr组合时，t_s 时间轴要能包含所有被试hr的时间范围。生成的图命名为 {win_id}_hr_comparison.png
# 2. RR 原始信号图：每个被试在同一窗口上绘制独立的rr原始信号图，原始RR图须展示被试窗口内RR数据波动全貌，Y 轴范围根据数据调整。时间轴 t_s 做去基线处理 df['t_s_relative'] = df['t_s'] - df['t_s'].min()，然后使用这个 t_s_relative 来绘图，这样所有被试的曲线都会从 0 秒开始，更便于比较。生成的图命名为 {win_id}_{sub_id})_rr.png

# 要绘制的被试ID列表与窗口ID列表，按照这个列表生成预览图
SID = [
    # "P001S001T001R001",
    # "P002S001T002R001",
    # "P003S001T001R001",
    # "P004S001T002R001",
    # "P006S001T002R001",
    # "P007S001T001R001",
    # "P008S001T002R001",
    # "P009S001T001R001",
    # "P010S001T002R001",
    # "P011S001T001R001",
    # "P012S001T001R001",
    # "P013S001T002R001",
    # "P014S001T001R001",
    # "P015S001T002R001",
    # "P016S001T001R001",
    # "P017S001T001R001",
    # "P018S001T001R001",
    # "P019S001T001R001",
    # "P020S001T001R001",
    # "P021S001T001R001",
    # "P022S001T001R001",
    # "P023S001T001R001",
    # "P024S001T002R001",
    # "P025S001T002R001",
    # "P026S001T002R001",
    # "P027S001T002R001",
    # "P028S001T002R001",
    # "P029S001T002R001",
    # "P030S001T002R001",
    # "P031S001T002R001",
    # "P032S001T001R001",
    # "P033S001T001R001",
    # "P034S001T001R001",
    # "P035S001T002R001",
    "P036S001T002R001",
    # "P037S001T002R001",
    # "P038S001T001R001",
    ]
WIN = ['w01','w02','w03','w04','w05','w06']
WIN = ['w05']
SIG = ["rr"]          # 只画呼吸相关（跨窗 RESP 概图）
# SIG = ["rr"]          # 只画 RR/HR 相关
# SIG = ["ecg"]         # 只画 ECG zoom
# SIG = ["rr","resp"]   # 两类都画
# SIG = []              # 空 = 默认全画
import matplotlib.pyplot as plt

# ----------- RESP cleaning summary helpers -----------
import ast

def _find_clean_resp_summary_csv() -> Path | None:
    """Find the latest clean_resp_summary*.csv under SRC_ROOT/collected (and SRC_ROOT as fallback)."""
    cands: list[Path] = []
    # common locations
    cands += sorted((SRC_ROOT / "collected").glob("clean_resp_summary*.csv"))
    cands += sorted(SRC_ROOT.glob("clean_resp_summary*.csv"))
    if not cands:
        return None
    # pick the most recently modified
    cands = [p.resolve() for p in cands]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _load_clean_resp_summary_map() -> dict[tuple[str, str], dict]:
    """Load clean_resp_summary csv and build a map keyed by (sid, win_id).

    Expected columns: subject_id, final_order (or w_id), status, kept_spans.
    We tolerate a few column name variants and parse kept_spans via ast.literal_eval.
    """
    path = _find_clean_resp_summary_csv()
    if path is None or (not path.exists()):
        print("[warn] clean_resp_summary not found; RESP plots will not be filtered by cleaning status")
        return {}

    try:
        rep = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed to read clean_resp_summary: {path} ({e})")
        return {}

    # normalize column names
    cols = set(rep.columns)
    col_sid = "subject_id" if "subject_id" in cols else ("sid" if "sid" in cols else None)
    col_w = "final_order" if "final_order" in cols else ("w_id" if "w_id" in cols else None)
    col_status = "status" if "status" in cols else None
    col_kept = "kept_spans" if "kept_spans" in cols else None

    if col_sid is None or col_w is None or col_status is None or col_kept is None:
        print(f"[warn] clean_resp_summary missing required cols; got={list(rep.columns)}")
        return {}

    out: dict[tuple[str, str], dict] = {}

    def _w_to_win_id(v) -> str:
        # final_order might be 1..N; map to 'w01'..'wNN'
        try:
            iv = int(v)
            return f"w{iv:02d}"
        except Exception:
            s = str(v)
            # accept already formatted values
            m = re.match(r"w\d+", s)
            return s if m else s

    for _, r in rep.iterrows():
        sid = str(r[col_sid])
        win_id = _w_to_win_id(r[col_w])
        status = str(r[col_status])
        kept_raw = r[col_kept]
        kept: list[tuple[float, float]] = []
        if isinstance(kept_raw, str) and kept_raw.strip() not in ("", "[]", "nan", "None"):
            try:
                obj = ast.literal_eval(kept_raw)
                # expect list of tuples
                if isinstance(obj, list):
                    for it in obj:
                        if isinstance(it, (tuple, list)) and len(it) == 2:
                            t0 = float(it[0])
                            t1 = float(it[1])
                            kept.append((t0, t1))
            except Exception:
                kept = []
        out[(sid, win_id)] = {"status": status, "kept_spans": kept}

    print(f"[info] RESP plot filter loaded from {path.name} ({len(out)} rows)")
    return out


def _apply_kept_spans_to_resp_df(df0: pd.DataFrame, kept_spans: list[tuple[float, float]]) -> pd.DataFrame:
    """Set df0['value']=NaN outside kept_spans (kept_spans are on absolute 'time_s')."""
    if df0 is None or df0.empty:
        return df0
    if (not kept_spans) or ("time_s" not in df0.columns) or ("value" not in df0.columns):
        # nothing to keep => blank it out
        out = df0.copy()
        out["value"] = np.nan
        return out

    t = pd.to_numeric(df0["time_s"], errors="coerce").to_numpy(dtype=float)
    keep = np.zeros_like(t, dtype=bool)
    for t0, t1 in kept_spans:
        keep |= (t >= float(t0)) & (t <= float(t1))

    out = df0.copy()
    v = pd.to_numeric(out["value"], errors="coerce").to_numpy(dtype=float)
    v[~keep] = np.nan
    out["value"] = v
    # recompute relative time based on remaining time base (keep original window alignment: start at 0)
    # IMPORTANT: do not drop NaN values in 'value' here; keep gaps.
    if np.isfinite(out["time_s"]).any():
        out["t_s_relative"] = out["time_s"] - float(np.nanmin(out["time_s"]))
    return out

# -------------------- configurable plotting params --------------------
HR_YLIM = (50, 120)  # fixed y-limits for HR comparison plots
HR_LABEL = "HR (bpm)"
RR_LABEL = "RR interval (ms)"
T_LABEL_ABS = "Time (s)"
T_LABEL_REL = "Time since window start (s)"
GAP_SECONDS = 20.0  # gap between windows in per-subject RR overview (s)

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


# --------- overview RR plot across all windows for a subject ---------
def plot_rr_all_windows_for_subject(sid: str, win_ids: list[str]) -> None:
    """
    For a subject, plot RR tachogram across all windows, concatenated with a fixed gap.
    Each window's RR data is shifted in x by an accumulating offset plus GAP_SECONDS between windows.
    Vertical dashed lines and labels are drawn at window boundaries.
    Saves as {sid}_rr_across_windows.png in OUT_DIR.
    """
    segments = []
    window_offsets = []
    offset = 0.0
    rr_ranges = []
    for win_id in win_ids:
        path = _rr_path_for(sid, win_id)
        df = _load_rr_df(path)
        if df is None or df.empty:
            window_offsets.append(None)
            continue
        df = df.sort_values("t_s_relative").copy()
        x = df["t_s_relative"] + offset
        seg = df.copy()
        seg["x"] = x
        segments.append(seg)
        rr_ranges.append((seg["rr_ms"].min(), seg["rr_ms"].max()))
        window_offsets.append(offset)
        offset += seg["t_s_relative"].max() + GAP_SECONDS
    if not segments:
        print(f"[info] no RR data for subject {sid} across any window")
        return
    # Concatenate all segments
    all_df = pd.concat(segments, axis=0, ignore_index=True)
    # Determine y-limits with 5% padding
    ymin = float(min(r[0] for r in rr_ranges))
    ymax = float(max(r[1] for r in rr_ranges))
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 10.0
    y0, y1 = ymin - pad, ymax + pad
    # Calculate boundary midpoints between windows
    boundaries = []
    for i in range(len(window_offsets) - 1):
        o1 = window_offsets[i]
        o2 = window_offsets[i + 1]
        if o1 is not None and o2 is not None:
            # The gap is between end of window i and start of window i+1
            last_seg = segments[len(boundaries)]
            last_xmax = last_seg["x"].max()
            next_xmin = segments[len(boundaries)+1]["x"].min()
            mid = (last_xmax + next_xmin) / 2.0
            boundaries.append(mid)
        elif o1 is not None and o2 is None:
            # missing next window, skip
            continue
        elif o1 is None and o2 is not None:
            # missing prev window, skip
            continue
        # both None: both missing, skip
    # Plot
    fig, ax = plt.subplots(figsize=(20, 4.5))
    ax.scatter(all_df["x"], all_df["rr_ms"], s=8, alpha=0.8, linewidths=0, edgecolors="none")
    ax.set_xlabel(f"{T_LABEL_REL} (concatenated across windows)")
    ax.set_ylabel(RR_LABEL)
    ax.set_title(f"RR tachogram across windows | {sid}")
    ax.set_ylim(y0, y1)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    # Draw vertical lines and annotate at boundaries
    for mid in boundaries:
        ax.axvline(mid, color="gray", linestyle="--", linewidth=1.1, alpha=0.6)
        ax.text(mid, y1 - 0.04*(y1-y0), "window boundary", ha="center", va="top", rotation=90,
                color="gray", fontsize=10, alpha=0.8, backgroundcolor="white")
    fig.tight_layout()
    out_path = OUT_DIR / f"{sid}_rr_across_windows.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ok] saved {out_path}")


# --------- overview resp plot across all windows for a subject ---------

def _resp_path_for(sid: str, win_id: str) -> Path:
    """Build path to RESP file for a given subject and window.
    Expected pattern: {sid}_resp_{win_id}.(csv|parquet|pq)
    """
    # prefer csv, but accept parquet as well
    cand = [
        (SRC_DIR_RESP / f"{sid}_resp_{win_id}.csv").resolve(),
        (SRC_DIR_RESP / f"{sid}_resp_{win_id}.parquet").resolve(),
        (SRC_DIR_RESP / f"{sid}_resp_{win_id}.pq").resolve(),
    ]
    for p in cand:
        if p.exists():
            return p
    # last resort: glob anything matching
    hits = sorted(SRC_DIR_RESP.glob(f"{sid}_resp_{win_id}.*"))
    return hits[0].resolve() if hits else cand[0]


def _load_resp_df(path: Path) -> pd.DataFrame | None:
    """Load RESP file and normalize to columns: time_s, value.

    Notes:
      - User requested we only care about the first two columns (time/value); extra cols are ignored.
      - We try common column names first; if missing, we fall back to the first/second column.
      - Adds: t_s_relative (time rebased to the window start)
    """
    if not path.exists():
        print(f"[warn] missing RESP file: {path}")
        return None
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            df_raw = pd.read_parquet(path)
        else:
            df_raw = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed to read RESP {path}: {e}")
        return None

    if df_raw is None or df_raw.empty:
        print(f"[warn] empty RESP: {path}")
        return None

    cols = list(df_raw.columns)
    time_cols = ["time_s", "t_s", "time", "timestamp_s"]
    val_cols = ["value", "resp", "resp_raw", "signal"]
    col_time = next((c for c in time_cols if c in cols), None)
    col_val = next((c for c in val_cols if c in cols), None)

    # fallback: first two columns
    if col_time is None:
        col_time = cols[0]
    if col_val is None:
        col_val = cols[1] if len(cols) >= 2 else cols[0]

    df = df_raw[[col_time, col_val]].rename(columns={col_time: "time_s", col_val: "value"}).copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # IMPORTANT: keep NaN in value to preserve gaps created by clean_resp
    df = df.dropna(subset=["time_s"]).sort_values("time_s")
    if df.empty:
        print(f"[warn] empty after numeric coercion: {path}")
        return None

    df["t_s_relative"] = df["time_s"] - float(df["time_s"].min())
    return df


def _smooth_series_rolling_median(x: np.ndarray, win: int = 5) -> np.ndarray:
    """Rolling-median smoothing that preserves NaN gaps (does NOT inpaint)."""
    if x.size == 0:
        return x
    w = int(max(1, win))
    s = pd.Series(x)
    y = s.rolling(window=w, center=True, min_periods=1).median().to_numpy(dtype=float)
    # preserve NaN positions (avoid filling gaps)
    y[~np.isfinite(x)] = np.nan
    return y


def _agg_value_per_hz(df: pd.DataFrame, hz: float = 1.0) -> pd.DataFrame:
    """Aggregate raw 'value' by median on a fixed-rate time grid (hz).

    Key: we keep a *complete* integer bin grid so that gaps remain gaps (NaN),
    which prevents fake slopes/bridges in plots and peak-based BR.
    """
    if df is None or df.empty:
        return df

    ts = pd.to_numeric(df["t_s_relative"].values, errors="coerce")
    vals = pd.to_numeric(df["value"].values, errors="coerce")

    # keep NaN in vals; only require time to be finite
    m = np.isfinite(ts)
    ts = ts[m]
    vals = vals[m]
    if ts.size == 0:
        return pd.DataFrame(columns=["t_s_relative", "value"])

    hz = float(hz)
    bins = np.floor(ts * hz).astype(np.int64)

    # group median (NaN-only bins will stay NaN if present)
    g = pd.DataFrame({"bin": bins, "value": vals}).groupby("bin", as_index=True)["value"].median()

    # reindex to full bin range to preserve gaps
    b0 = int(np.nanmin(bins))
    b1 = int(np.nanmax(bins))
    full = pd.RangeIndex(b0, b1 + 1)
    g = g.reindex(full)

    out = g.reset_index().rename(columns={"index": "bin", "value": "value"})
    out["t_s_relative"] = out["bin"].astype(float) / hz
    return out[["t_s_relative", "value"]]


def _find_peaks_robust(y: np.ndarray, distance: int, prominence: float, width: int | None = None) -> np.ndarray:
    """Robust peak finder: use SciPy if available, else a conservative fallback."""
    try:
        from scipy.signal import find_peaks  # type: ignore
        kwargs = {"distance": int(max(1, distance)), "prominence": float(max(0.0, prominence))}
        if width is not None:
            kwargs["width"] = int(max(1, width))
        idx, _ = find_peaks(y, **kwargs)
        return np.asarray(idx, dtype=int)
    except Exception:
        # Fallback: local maxima with minimal distance and simple prominence-like check
        if y.size < 3:
            return np.array([], dtype=int)
        dy_prev = np.r_[np.nan, np.diff(y)]
        dy_next = np.r_[np.diff(y), np.nan]
        cand = np.where((dy_prev > 0) & (dy_next < 0))[0]
        if cand.size == 0:
            return np.array([], dtype=int)
        # score by height
        cand = cand[np.argsort(y[cand])[::-1]]
        kept: list[int] = []
        for i in cand:
            if any(abs(i - k) < distance for k in kept):
                continue
            left = max(0, i - distance)
            right = min(len(y) - 1, i + distance)
            base = min(np.nanmin(y[left:i+1]) if i > left else y[i],
                       np.nanmin(y[i:right+1]) if right > i else y[i])
            if (y[i] - base) >= prominence:
                kept.append(int(i))
        kept.sort()
        return np.asarray(kept, dtype=int)




def _contiguous_true_spans(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """Return list of (start, end) slices (end exclusive) where mask is True contiguously."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(mask.tolist()):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            if (i - start) >= int(min_len):
                spans.append((start, i))
            start = None
    if start is not None:
        if (len(mask) - start) >= int(min_len):
            spans.append((start, len(mask)))
    return spans


def _compute_resp_br_series_from_window_df(df_raw: pd.DataFrame, rate_hz: float = 5.0, resp_max_bpm: float = 30.0) -> pd.DataFrame | None:
    """Compute instantaneous breathing rate series for ONE window (time is t_s_relative).

    Important: if the cleaned RESP contains gaps (NaN), we MUST NOT bridge them.
    We therefore (1) downsample onto a complete fixed-rate grid with NaNs preserved,
    (2) split into contiguous finite spans, and (3) compute peaks/BR within each span.

    Returns DataFrame with columns: t_s_relative, br_bpm.
    """
    if df_raw is None or df_raw.empty:
        return None

    rate_hz = float(rate_hz)

    # downsample to fixed-rate grid; NaN gaps preserved
    df = _agg_value_per_hz(df_raw, hz=rate_hz)
    if df is None or df.empty or df.shape[0] < 5:
        return None

    y_full = df["value"].to_numpy(dtype=float)
    t_full = df["t_s_relative"].to_numpy(dtype=float)

    finite = np.isfinite(y_full) & np.isfinite(t_full)
    # require at least ~2s continuous data per span (or 10 points)
    min_span_len = int(max(10, round(rate_hz * 2.0)))
    spans = _contiguous_true_spans(finite, min_len=min_span_len)
    if not spans:
        return None

    pieces: list[pd.DataFrame] = []

    for s, e in spans:
        y = y_full[s:e].astype(float)
        t = t_full[s:e].astype(float)
        if y.size < 5:
            continue

        # smooth within span only (no NaNs inside)
        y = _smooth_series_rolling_median(y, win=5)
        k = max(3, int(round(rate_hz * 1.0)))  # ~1s mean smoothing
        y = pd.Series(y).rolling(window=k, center=True, min_periods=1).mean().to_numpy(dtype=float)

        if y.size < 5 or not np.isfinite(y).any():
            continue

        q1, q99 = np.nanpercentile(y, [1, 99])
        prom = max(1e-6, 0.15 * float(q99 - q1))

        alpha = 0.9
        min_dist = int(np.ceil(rate_hz * (60.0 / float(resp_max_bpm)) * alpha))
        width_pts = int(max(1, round(rate_hz * 0.3)))

        idx = _find_peaks_robust(y, distance=min_dist, prominence=prom, width=width_pts)
        if idx.size < 3:
            continue

        t_peaks = t[idx]
        ibi = np.diff(t_peaks)
        t_mid = (t_peaks[1:] + t_peaks[:-1]) / 2.0
        br = 60.0 / np.maximum(ibi, 1e-6)
        br_s = pd.Series(br).rolling(window=3, center=True, min_periods=1).median().to_numpy(dtype=float)

        pieces.append(pd.DataFrame({"t_s_relative": t_mid.astype(float), "br_bpm": br_s.astype(float)}))

    if not pieces:
        return None

    out = pd.concat(pieces, axis=0, ignore_index=True).sort_values("t_s_relative")
    return out


def plot_resp_all_windows_for_subject(sid: str, win_ids: list[str]) -> None:
    """Concatenate RESP across windows with a fixed gap and plot a 2-panel overview.

    Top: raw RESP (units preserved), concatenated in time.
    Bottom: breathing rate (breaths/min) derived from peak intervals (robust), concatenated in time.

    Output: {sid}_resp_across_windows.png in OUT_DIR.
    """
    summary_map = _load_clean_resp_summary_map()
    segments_raw: list[pd.DataFrame] = []
    segments_br: list[pd.DataFrame] = []
    window_offsets: list[float | None] = []

    offset = 0.0
    y_ranges = []

    # parameters (match your other plotting logic)
    resp_max_bpm = float(PARAMS.get("resp_max_bpm", 30.0))
    raw_plot_hz = float(PARAMS.get("resp_plot_raw_hz", 2.0))

    for win_id in win_ids:
        path = _resp_path_for(sid, win_id)
        # Honor clean_resp_summary: skip rejected windows or windows with empty kept_spans.
        meta = summary_map.get((sid, win_id), None)
        if meta is not None:
            status = str(meta.get("status", ""))
            kept_spans = meta.get("kept_spans", [])
            if (status.lower() == "reject") or (not kept_spans):
                window_offsets.append(None)
                continue
        else:
            kept_spans = []

        df0 = _load_resp_df(path)
        if df0 is None or df0.empty:
            window_offsets.append(None)
            continue

        # If we have kept_spans from the report, mask out everything else as NaN.
        if meta is not None:
            df0 = _apply_kept_spans_to_resp_df(df0, kept_spans)
            # If everything became NaN after masking, treat as missing.
            if ("value" in df0.columns) and (not np.isfinite(df0["value"].to_numpy(dtype=float)).any()):
                window_offsets.append(None)
                continue

        # readability: if very long, downsample to ~raw_plot_hz (default 2 Hz)
        dfp = df0.copy()
        if dfp.shape[0] > 5000:
            dfp = _agg_value_per_hz(dfp, hz=raw_plot_hz)
        # light smoothing for the raw curve (readability only)
        dfp["value"] = _smooth_series_rolling_median(dfp["value"].to_numpy(dtype=float), win=5)

        dfp = dfp.sort_values("t_s_relative").copy()
        dfp["x"] = dfp["t_s_relative"] + offset
        segments_raw.append(dfp[["x", "value"]].copy())
        v = dfp["value"].to_numpy(dtype=float)
        if np.isfinite(v).any():
            y_ranges.append((float(np.nanmin(v)), float(np.nanmax(v))))

        # breathing rate series (uses 5 Hz internally)
        br_df = _compute_resp_br_series_from_window_df(df0, rate_hz=5.0, resp_max_bpm=resp_max_bpm)
        if br_df is not None and not br_df.empty:
            br_df = br_df.sort_values("t_s_relative").copy()
            br_df["x"] = br_df["t_s_relative"] + offset
            segments_br.append(br_df[["x", "br_bpm"]].copy())

        window_offsets.append(offset)
        offset += float(dfp["t_s_relative"].max()) + GAP_SECONDS

    if not segments_raw:
        print(f"[info] no RESP data for subject {sid} across any window")
        return

    # y-limits for raw with padding (ignore all-NaN windows)
    if y_ranges:
        ymin = float(min(r[0] for r in y_ranges))
        ymax = float(max(r[1] for r in y_ranges))
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
        y0, y1 = ymin - pad, ymax + pad
    else:
        # fallback if everything is NaN
        y0, y1 = 0.0, 1.0

    # boundary positions: for each *present* window, boundary is at end_x + GAP/2
    # (this avoids fragile indexing when some windows are missing)
    end_x_list: list[float] = []
    for seg in segments_raw:
        end_x_list.append(float(seg["x"].max()))
    boundaries: list[float] = [ex + (GAP_SECONDS / 2.0) for ex in end_x_list[:-1]]

    # plot
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 7.0), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # top: raw resp (plot per-window segments to prevent cross-window line bridging)
    for seg in segments_raw:
        ax0.plot(seg["x"], seg["value"], linewidth=1.0, alpha=0.9)
    ax0.set_ylabel("RESP (raw units)")
    ax0.set_ylim(y0, y1)
    ax0.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    # bottom: BR (plot per-window pieces; BR is already gap-safe within each window)
    if segments_br:
        for seg in segments_br:
            ax1.plot(seg["x"], seg["br_bpm"], linewidth=1.2, alpha=0.95)
        ax1.set_ylabel("BR (breaths/min)")
        ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    else:
        ax1.text(0.5, 0.5, "too few peaks for BR", transform=ax1.transAxes,
                 ha="center", va="center", color="gray")
        ax1.set_yticks([])
        ax1.grid(False)

    ax1.set_xlabel(f"{T_LABEL_REL} (concatenated across windows)")

    # boundaries and labels
    for mid in boundaries:
        ax0.axvline(mid, color="gray", linestyle="--", linewidth=1.1, alpha=0.6)
        ax1.axvline(mid, color="gray", linestyle="--", linewidth=1.1, alpha=0.6)
        ax0.text(mid, y1 - 0.04 * (y1 - y0), "window boundary", ha="center", va="top",
                 rotation=90, color="gray", fontsize=10, alpha=0.8, backgroundcolor="white")

    fig.suptitle(f"RESP across windows | {sid}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / f"{sid}_resp_across_windows.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ok] saved {out_path}")

# --------- overview resp plot across all windows for a subject ---------
# 请按照类似 rr plot across windows 的方法绘制一张 resp 图。
# 概图把几个窗口的 resp 拼在一张图中，中间用 "window boundary" 区分边界。
# 图分上下两部分，上面是原始 resp 信号图，下面是 plot_resp_interval 那样绘制一张曲线图表示呼吸频率变化。
# 读取resp数据路径 SRC_DIR_RESP，已经写到本脚本的全局变量里。输出数据放在 OUT_DIR 
# 概图要参考 全局变量 SID ，SID 列表中包含被试才会绘制，如果该列表为空绘制所有被试
# 最后在main里来 执行这个函数

# ecg plot
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
    # SIG filter: only plot requested signals; keep default as "all" if SIG is empty/None
    sig_set = {s.strip().lower() for s in (SIG or []) if isinstance(s, str) and s.strip()}
    if not sig_set:
        sig_set = {"rr", "ecg", "resp"}

    # loop over requested windows
    for win_id in WIN:
        print(f"\n[window] {win_id}")

        if "rr" in sig_set:
            plot_hr_comparison_for_window(win_id, SID)
            plot_rr_per_subject_for_window(win_id, SID)
        else:
            print("[skip] rr plots (SIG filter)")

        if "ecg" in sig_set:
            plot_ecg_per_subject_for_window(win_id, SID)
        else:
            print("[skip] ecg plots (SIG filter)")

    if "rr" in sig_set:
        print("\n[subject-level RR overview]")
        for sid in SID:
            plot_rr_all_windows_for_subject(sid, WIN)
    else:
        print("\n[skip] subject-level RR overview (SIG filter)")

    if "resp" in sig_set:
        print("\n[subject-level RESP overview]")
        for sid in SID:
            plot_resp_all_windows_for_subject(sid, WIN)
    else:
        print("\n[skip] subject-level RESP overview (SIG filter)")

if __name__ == "__main__":
    main()