import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
# optional peak finder (fallback to numpy if unavailable)
try:
    from scipy.signal import find_peaks as _scipy_find_peaks
except Exception:
    _scipy_find_peaks = None

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATASETS, ACTIVE_DATA, DATA_DIR, SCHEMA

# Dataset config
DS = DATASETS[ACTIVE_DATA]
paths = DS["paths"]
SRC_DIR = (DATA_DIR / paths["confirmed"] ).resolve()
SRC_NORM_DIR = (DATA_DIR / paths["norm"]).resolve()
OUT_ROOT = (DATA_DIR / paths["preview"] ).resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PRE_SID: List = DS["preview_sids"] # 选择 PRE_SID[0]
# PRE_SIG: List = DS["windowing"]["apply_to"] # 选择列表中的所有信号
PRE_SIG: List =["rr"] # 选择列表中的所有信号

FILE_TYPES = ["csv", "parquet"]

# --- plotting ranges (can be tuned):
# Force y-limits for RR raw plot in milliseconds. Set to None to disable forced limits.
RR_RAW_FORCE_YLIM = (300, 1200)  # e.g., set to (0, 1000) if you prefer that view
# Force y-limits for 1 Hz HR plot in bpm. Set to None to disable forced limits.
HR1HZ_FORCE_YLIM = (40, 120)
# 呼吸频率图的幅度控制
# Force y-limits for breathing rate (BR, breaths/min). None to disable.
BR_FORCE_YLIM = (6, 30)
# 绘制呼吸图时曲线的最小间距，控制稀疏程度。越大越稀疏
RESP_MAX_BPM = 30
# ecg 放大图的起点（make_windowing_ecg_plot），事件名或具体时间点选其一
ECG_PlOT_START_EVENT = None
ECG_PLOT_START = 88070
# ecg 放大图时窗口的大小，
ECG_PLOT_SPAN = 50

# 不按照系统设定，临时检查文件
# SRC_DIR = (DATA_DIR / paths["confirmed"])
PRE_SID = [
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
    "P039S001T001R001",
    "P040S001T001R001",
    "P041S001T001R001",
    "P042S001T001R001",
    "P043S001T001R001",
    "P044S001T001R001",
    "P045S001T001R001",
    "P046S001T001R001",
    "P047S001T001R001",
    "P048S001T001R001",
    "P049S001T002R001",
    "P050S001T002R001",
    "P051S001T002R001",
    "P052S001T002R001",
    "P053S001T002R001",
    "P054S001T002R001",
    "P055S001T002R001",
    "P056S001T002R001",
    "P057S001T002R001",
    "P058S001T002R001",
    "P059S001T002R001",
    "P060S001T002R001",
    "P061S001T002R001",
    "P062S001T002R001",
    "P063S001T001R001",
    "P064S001T001R001",
    "P065S001T001R001",
    "P066S001T001R001",
    ]
PRE_SIG = ["rr"] # "rr", "ecg", "resp"

PRE_SID = [
    # "P060S001T002R001",
    # "P061S001T002R001",
    "P012S001T001R001",
    # "P064S001T001R001",
    # "P065S001T001R001",
    # "P066S001T001R001",
    ]
PRE_SIG = ["rr"]

# Use SCHEMA-defined canonical column names first, then fallback heuristics
def _get_time_col(df: pd.DataFrame, signal: str) -> Optional[str]:
    spec = SCHEMA.get(signal, {})
    tcol = spec.get("t")
    if tcol and tcol in df.columns:
        return tcol
    for c in ("t_s", "time_s"):
        if c in df.columns:
            return c
    # last resort: any monotonic numeric column
    for c in df.columns:
        if df[c].dtype.kind in "fi" and df[c].is_monotonic_increasing:
            return c
    return None

def _choose_value_series(df: pd.DataFrame, signal: str, tcol: str) -> Optional[pd.Series]:
    spec = SCHEMA.get(signal, {})
    # 3-axis acceleration: prefer magnitude from vx,vy,vz; fallback to any available axis
    if signal == "acc":
        vx = spec.get("vx", "value_x")
        vy = spec.get("vy", "value_y")
        vz = spec.get("vz", "value_z")
        cols = [c for c in (vx, vy, vz) if c in df.columns]
        if len(cols) >= 2:
            mag = None
            for c in cols:
                v = pd.to_numeric(df[c], errors="coerce")
                mag = v**2 if mag is None else mag + v**2
            mag = np.sqrt(mag)
            return mag.rename("acc_magnitude")
        if len(cols) == 1:
            return pd.to_numeric(df[cols[0]], errors="coerce").rename(cols[0])
    # generic continuous signals (resp/ecg/ppg/hr/etc.)
    vcol = spec.get("v")
    if vcol and vcol in df.columns:
        # RR special-case: derive HR if rr_ms/ibi present
        if signal == "rr":
            return _maybe_compute_hr(df, vcol, signal)
        return pd.to_numeric(df[vcol], errors="coerce")
    # fallback: first numeric column excluding time
    for c in df.columns:
        if c == tcol:
            continue
        if df[c].dtype.kind in "fi":
            return pd.to_numeric(df[c], errors="coerce")
    return None

def _read_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None

def _find_signal_file(src_dir: Path, sid: str, signal: str) -> Optional[Path]:
    # try parquet first, then csv
    pats = [f"{sid}_{signal}.parquet", f"{sid}_{signal}.csv",
            f"{sid}_{signal}*.parquet", f"{sid}_{signal}*.csv"]
    for pat in pats:
        hits = list(src_dir.glob(pat))
        if hits:
            return sorted(hits)[0]
    return None

def _maybe_compute_hr(df: pd.DataFrame, vcol: str, signal: str) -> pd.Series:
    # If RR present as ms, derive HR for interpretability; otherwise use original
    series = df[vcol]
    if signal == "rr":
        name = vcol.lower()
        if "rr" in name and "ms" in name:
            with np.errstate(divide="ignore", invalid="ignore"):
                hr = 60000.0 / series.astype(float)
                return hr.rename("hr_from_rr")
    return series.astype(float)

def _robust_minmax(x: pd.Series) -> pd.Series:
    # clip to 1%~99% then scale to 0..1; if degenerate fall back to z-score
    x = pd.to_numeric(x, errors="coerce")
    q1, q99 = x.quantile(0.01), x.quantile(0.99)
    x_clip = x.clip(q1, q99)
    rng = float(q99 - q1)
    if rng <= 0 or not np.isfinite(rng):
        mu, sd = x.mean(), x.std(ddof=0)
        if sd and np.isfinite(sd):
            z = (x - mu) / sd
            z = (z - z.min()) / (z.max() - z.min() + 1e-12)
            return z
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x_clip - q1) / (rng + 1e-12)

def load_signal_norm(src_dir: Path, sid: str, signal: str) -> Optional[pd.DataFrame]:
    fp = _find_signal_file(src_dir, sid, signal)
    if fp is None:
        print(f"[warn] {sid}: no file for signal '{signal}' under {src_dir}")
        return None
    df = _read_table(fp)
    if df is None or df.empty:
        print(f"[warn] {sid}: failed reading {fp}")
        return None
    tcol = _get_time_col(df, signal)
    if tcol is None:
        print(f"[warn] {sid}: cannot find time column for '{signal}' in {fp}")
        return None
    vals = _choose_value_series(df, signal, tcol)
    if vals is None:
        print(f"[warn] {sid}: cannot find value series for '{signal}' in {fp}")
        return None
    norm = _robust_minmax(vals)
    out = pd.DataFrame({"time_s": df[tcol].astype(float), "norm": norm.astype(float)})
    out = out.dropna(subset=["time_s","norm"])
    return out

def load_events(src_dir: Path, sid: str) -> Optional[pd.DataFrame]:
    """Load events for a subject from norm/ and return standardized columns: time_s, label.
    If not found or unreadable, return None.
    """
    fp = _find_signal_file(src_dir, sid, "events")
    if fp is None:
        print(f"[info] {sid}: no events file under {src_dir}")
        return None
    df = _read_table(fp)
    if df is None or df.empty:
        print(f"[warn] {sid}: failed reading events file {fp}")
        return None
    spec = SCHEMA.get("events", {})
    t_key = spec.get("t", "time_s")
    label_key = spec.get("label", None) or ("events" if "events" in df.columns else "label")
    # Fallbacks
    if t_key not in df.columns:
        t_key = "time_s" if "time_s" in df.columns else df.columns[0]
    if label_key not in df.columns:
        if "events" in df.columns:
            label_key = "events"
        elif "label" in df.columns:
            label_key = "label"
        else:
            # take the last column as label as last resort
            label_key = df.columns[-1]
    out = pd.DataFrame({
        "time_s": pd.to_numeric(df[t_key], errors="coerce"),
        "label": df[label_key].astype(str)
    }).dropna(subset=["time_s"]) 
    return out

def _agg_per_hz(df, hz=1.0):
    # 把 time_s 按 hz 聚到等间隔网格，聚合用中位数，返回 time_s(秒格)与 norm
    # 使用此方法避免线条
    if df is None or df.empty:
        return df
    ts = pd.to_numeric(df["time_s"].values, errors="coerce")
    bins = np.floor(ts * hz).astype(np.int64)  # 每秒/半秒为一个bin
    g = pd.DataFrame({"bin": bins, "norm": df["norm"].values}).groupby("bin", as_index=False)["norm"].median()
    g["time_s"] = g["bin"] / hz
    return g[["time_s","norm"]]

def _smooth(df, win=3):
    """Median smoothing on the 'norm' column to improve readability.
    Keeps time points unchanged; no effect if df is empty or win<=1.
    """
    if df is None or df.empty or win <= 1:
        return df
    y = pd.Series(df["norm"].values).rolling(window=win, center=True, min_periods=1).median()
    out = df.copy()
    out["norm"] = y.values
    return out

def _agg_value_per_hz(df, hz=1.0):
    """Aggregate raw 'value' by median on a fixed-rate time grid (hz).
    Keeps units unchanged; suitable for readability in raw-scale plots.
    """
    if df is None or df.empty:
        return df
    ts = pd.to_numeric(df["time_s"].values, errors="coerce")
    vals = pd.to_numeric(df["value"].values, errors="coerce")
    bins = np.floor(ts * hz).astype(np.int64)
    g = pd.DataFrame({"bin": bins, "value": vals}).groupby("bin", as_index=False)["value"].median()
    g["time_s"] = g["bin"] / hz
    return g[["time_s", "value"]]

# --- Helper: median smoothing for raw value plotting ---

def _smooth_value(df, win=3):
    """Median smoothing on the 'value' column (raw-scale plotting).
    Keeps time points and units unchanged; no effect if df is empty or win<=1.
    """
    if df is None or df.empty or win <= 1 or "value" not in df.columns:
        return df
    y = pd.Series(df["value"].values).rolling(window=win, center=True, min_periods=1).median()
    out = df.copy()
    out["value"] = y.values
    return out

# --- Helper: robust peak finding for 1D array, with scipy or numpy fallback ---
def _find_peaks_robust(y: np.ndarray, distance: int, prominence: float, width: Optional[int] = None) -> np.ndarray:
    """Return indices of peaks in 1D array y.
    Prefers scipy.signal.find_peaks if available; otherwise uses a numpy fallback.
    `distance` is the minimal number of samples between adjacent peaks.
    `prominence` is the minimal vertical prominence (in same units as y).
    `width` is the minimal width (in samples) of peaks; ignored in numpy fallback.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.array([], dtype=int)
    # Try SciPy if available
    if _scipy_find_peaks is not None:
        try:
            kw = dict(distance=max(1, int(distance)), prominence=max(0.0, float(prominence)))
            if width is not None:
                kw["width"] = max(1, int(width))
            idx, props = _scipy_find_peaks(y, **kw)
            return idx.astype(int)
        except Exception:
            pass
    # Numpy fallback: simple local maxima with distance & prominence filtering (width ignored)
    dy_prev = np.r_[np.nan, np.diff(y)]
    dy_next = np.r_[np.diff(y), np.nan]
    cand = np.where((dy_prev > 0) & (dy_next < 0))[0]
    if cand.size == 0:
        return cand
    cand = cand[np.argsort(y[cand])[::-1]]  # sort by height desc
    kept = []
    for i in cand:
        if any(abs(i - k) < distance for k in kept):
            continue
        left = max(0, i - int(distance))
        right = min(len(y) - 1, i + int(distance))
        base = min(np.nanmin(y[left:i+1]) if i > left else y[i], np.nanmin(y[i:right+1]) if right > i else y[i])
        if (y[i] - base) >= prominence:
            kept.append(i)
    kept.sort()
    return np.array(kept, dtype=int)

def _prep_resp_for_plot(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Preprocess RESP for raw plotting: aggregate, smooth, and return (df_plot, hz).
    We keep ORIGINAL units, just make it readable.
    """
    # Aggregate to ~2 Hz for readability if very long
    hz = 2.0
    df_plot = _agg_value_per_hz(df_raw, hz=hz) if len(df_raw) > 5000 else df_raw.copy()
    # 2-step smoothing: median then mean over ~1.5–2 s
    df_plot = _smooth_value(df_plot, win=5)
    k = max(3, int(hz * 2.0))  # ~2 s mean window
    s = pd.Series(df_plot["value"].values).rolling(window=k, center=True, min_periods=1).mean()
    df_plot["value"] = s.values
    return df_plot, hz

# --- Helper: overlay events on axes ---
def _overlay_events_on_axes(ax, sid: str):
    """Overlay events on given axes: red dashed vertical lines + labels.
    Uses events from norm/; prints info if none found.
    """
    ev = load_events(SRC_NORM_DIR, sid)
    if ev is not None and not ev.empty:
        ymin, ymax = ax.get_ylim()
        for _, row in ev.iterrows():
            x = float(row["time_s"]) if pd.notna(row["time_s"]) else None
            if x is None:
                continue
            ax.axvline(x=x, color="red", linestyle="--", linewidth=0.8, alpha=0.85, zorder=3)
            lbl = str(row.get("label", ""))
            if lbl:
                ax.text(x, ymax, lbl, rotation=90, va="top", ha="right", fontsize=8, color="red",
                        alpha=0.9, zorder=3, clip_on=True)
    else:
        print(f"[info] {sid}: no events to annotate under norm")


# --- Helper: overlay events as lines only (no labels, for stacked plots) ---
def _overlay_events_lines(ax, sid: str):
    """Overlay events as vertical red dashed lines WITHOUT labels (to reduce clutter on stacked plots)."""
    ev = load_events(SRC_NORM_DIR, sid)
    if ev is None or ev.empty:
        return
    for _, row in ev.iterrows():
        x = float(row["time_s"]) if pd.notna(row["time_s"]) else None
        if x is None:
            continue
        ax.axvline(x=x, color="red", linestyle="--", linewidth=0.8, alpha=0.55, zorder=2)

def load_signal_raw(src_dir: Path, sid: str, signal: str) -> Optional[tuple[pd.DataFrame, str]]:
    """Load a signal in ORIGINAL scale (no normalization, no RR→HR conversion).
    Returns (df, value_name) where df has columns: time_s, value.
    For RR, returns rr_ms if available; ACC is not intended for raw plotting here.
    """
    if signal == "acc":
        return None
    fp = _find_signal_file(src_dir, sid, signal)
    if fp is None:
        print(f"[warn] {sid}: no file for signal '{signal}' under {src_dir}")
        return None
    df = _read_table(fp)
    if df is None or df.empty:
        print(f"[warn] {sid}: failed reading {fp}")
        return None
    tcol = _get_time_col(df, signal)
    if tcol is None:
        print(f"[warn] {sid}: cannot find time column for '{signal}' in {fp}")
        return None
    # choose value column in ORIGINAL units
    spec = SCHEMA.get(signal, {})
    vcol = spec.get("v", None)
    if signal == "rr":
        # Prefer rr_ms/ibi in milliseconds if present
        candidates = [c for c in df.columns if c.lower() in ("rr_ms")]
        if vcol and vcol in df.columns:
            candidates.insert(0, vcol)
        vcol = None
        for c in candidates:
            if c in df.columns:
                vcol = c; break
        if vcol is None:
            # fallback: first numeric not time
            for c in df.columns:
                if c == tcol: continue
                if df[c].dtype.kind in "fi":
                    vcol = c; break
        ylabel = vcol or "value"
    else:
        if not vcol or vcol not in df.columns:
            # fallback: first numeric not time
            vcol = None
            for c in df.columns:
                if c == tcol: continue
                if df[c].dtype.kind in "fi":
                    vcol = c; break
        ylabel = vcol or "value"
    vals = pd.to_numeric(df[vcol], errors="coerce") if vcol else None
    if vals is None:
        print(f"[warn] {sid}: cannot find value series for '{signal}' in {fp}")
        return None
    out = pd.DataFrame({
        "time_s": pd.to_numeric(df[tcol], errors="coerce"),
        "value": vals
    }).dropna(subset=["time_s", "value"]) 
    return out, ylabel

# --- ecg 图 ---
def _plot_ecg_signal(sid: str, sig: str):
    """
    绘制 ecg 信号
    """
    src = SRC_NORM_DIR
    loaded = load_signal_raw(src, sid, sig)
    if not loaded:
        return
    df_raw, ylabel = loaded
    if df_raw is None or df_raw.empty:
        print(f"[warn] {sid}: empty {sig} in ECG plot")
        return

    # Aggregate to 50 Hz for readability if long, else keep as is
    if len(df_raw) > 5000:
        df_plot = _agg_value_per_hz(df_raw, hz=50.0)
    else:
        df_plot = df_raw.copy()

    # Crop to event window if events exist
    ev = load_events(SRC_NORM_DIR, sid)
    if ev is not None and not ev.empty:
        t0 = float(np.nanmin(ev["time_s"])) - 5.0
        t1 = float(np.nanmax(ev["time_s"])) + 5.0
        df_plot = df_plot[(df_plot["time_s"] >= t0) & (df_plot["time_s"] <= t1)]

    # Light median smoothing (win=3)
    df_plot = _smooth_value(df_plot, win=3)

    # Robust y-limits from percentiles (units preserved)
    if df_plot["value"].notna().sum() >= 5:
        q1, q99 = np.nanpercentile(df_plot["value"], [1, 99])
        span = max(1e-9, q99 - q1)
        ymin, ymax = q1 - 0.1 * span, q99 + 0.1 * span
    else:
        ymin, ymax = None, None

    plt.figure(figsize=(14, 4))
    ax = plt.gca()
    ax.plot(df_plot["time_s"].values, df_plot["value"].values,
            linewidth=0.2, alpha=0.95, color="#4E79A7", label="ECG", zorder=2)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    _overlay_events_on_axes(ax, sid)
    ax.set_title(f"{sid} ECG (raw scale)", fontsize=14)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.2)

    out = (OUT_ROOT / f"{sid}_{sig}_raw.png").resolve()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[save] raw-{sig} → {out}")

def _plot_ecg_zoom_with_peaks_range(sid: str, t_start: float, span_s: float = 10.0):
    """
    诊断用：在 [t_start, t_start+span_s] 窗口内绘制 ECG，并自动标注 R 峰，并输出三行图：
    - 上：ECG（raw + 5–20Hz）+ R 峰
    - 中：基于 ECG 峰得到的 RR(ms)（含 QC 标注）
    - 下：device RR（H10 原生 RR）在同一时段的 RR(ms)
    - 自动估计采样率 fs
    - 轻带通滤波（5–20 Hz，突出 QRS；若 SciPy 不可用则跳过滤波）
    - 峰间最小距离由 200 bpm 推得（>= 0.3 s）
    - 峰的 prominence 由 1–99% 幅度自适应设定
    rr 图除了现有折线，还标注RR点。
    ecg-rr 图中 rr点为蓝色，连线为淡绿色，QC点标注红色
    device-rr图中，rr点为橙黄色，连线为淡粉色，无QC标注。
    """
    loaded = load_signal_raw(SRC_NORM_DIR, sid, "ecg")
    if not loaded:
        print(f"[warn] {sid}: no ECG for zoom"); return
    df_raw, ylabel = loaded
    if df_raw.empty:
        print(f"[warn] {sid}: empty ECG for zoom"); return

    t0 = float(t_start)
    t1 = float(t_start) + float(span_s)
    df = df_raw[(df_raw["time_s"] >= t0) & (df_raw["time_s"] <= t1)].copy()
    if df.shape[0] < 50:
        print(f"[info] {sid}: too few ECG points in the chosen window"); return

    import numpy as np
    import matplotlib.pyplot as plt

    t = df["time_s"].to_numpy(dtype=float)
    x = df["value"].to_numpy(dtype=float)

    # 估计采样率
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        print(f"[warn] {sid}: invalid time base"); return
    fs = 1.0 / dt

    # 带通（优先 SciPy；失败则用原波形）
    try:
        from scipy.signal import butter, filtfilt
        low, high = 5.0, 20.0
        nyq = 0.5 * fs
        b, a = butter(3, [low/nyq, high/nyq], btype="band")
        xf = filtfilt(b, a, x)
    except Exception:
        xf = x

    # 峰检测
    try:
        from scipy.signal import find_peaks
        use_scipy = True
    except Exception:
        use_scipy = False

    min_dist = int(max(1, np.round(0.30 * fs)))  # >=0.3s
    q1, q99 = np.nanpercentile(xf, [1, 99])
    prom = max(1e-3, 0.2 * (q99 - q1))

    if use_scipy:
        pk, _ = find_peaks(xf, distance=min_dist, prominence=prom)
    else:
        # 简单后备：局部极大值 + 距离/显著性过滤
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

    # RR(ms)
    rr_ms = None
    qc_flags = None
    tm = None
    if pk.size >= 2:
        rr_s = np.diff(t[pk])
        rr_ms = 1000.0 * rr_s
    # --- QC: flag suspicious RR without removing them ---
    if rr_ms is not None:
        rr = np.asarray(rr_ms, dtype=float)
        # length-based flags
        flags_len = (rr < 300.0) | (rr > 2000.0)
        # relative-change flags (to previous RR)
        rel = np.zeros_like(rr, dtype=bool)
        if rr.size >= 2:
            prev = rr[:-1]
            d = np.abs(np.diff(rr)) / np.maximum(prev, 1e-6)
            rel[1:] = d > 0.25  # >25% jump from previous interval
        qc_flags = flags_len | rel

        # prepare RR mid-time for plotting and stats
        tm = (t[pk][1:] + t[pk][:-1]) / 2.0

        # --- window statistics (ALL points) ---
        rr_med_all = float(np.nanmedian(rr)) if rr.size else np.nan
        hr_med_all = float(60000.0 / rr_med_all) if np.isfinite(rr_med_all) and rr_med_all > 0 else np.nan
        rmssd_all = float(np.sqrt(np.nanmean(np.diff(rr) ** 2))) if rr.size >= 2 else np.nan

        # --- window statistics (EXCLUDING QC-flagged), only if enough points ---
        if np.any(~qc_flags) and np.sum(~qc_flags) >= 2:
            rr_clean = rr[~qc_flags]
            rr_med_clean = float(np.nanmedian(rr_clean))
            hr_med_clean = float(60000.0 / rr_med_clean) if rr_med_clean > 0 else np.nan
            rmssd_clean = float(np.sqrt(np.nanmean(np.diff(rr_clean) ** 2))) if rr_clean.size >= 2 else np.nan
        else:
            rr_med_clean = np.nan
            hr_med_clean = np.nan
            rmssd_clean = np.nan

        # Print a concise summary for this window
        print(
            f"[stats] {sid} t=[{t0:.1f},{t1:.1f}]s | RR_med(all)={rr_med_all:.1f} ms, HR_med(all)={hr_med_all:.1f} bpm, RMSSD(all)={rmssd_all:.1f} ms; "
            f"RR_med(clean)={rr_med_clean:.1f} ms, HR_med(clean)={hr_med_clean:.1f} bpm, RMSSD(clean)={rmssd_clean:.1f} ms"
        )

    # 画图
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1, 1]})

    ax0.plot(t, x, color="#a6bddb", linewidth=0.6, alpha=0.8, label="ECG raw")
    ax0.plot(t, xf, color="#225ea8", linewidth=0.8, alpha=0.9, label="ECG (5–20 Hz)")
    if pk.size:
        ax0.scatter(t[pk], xf[pk], color="crimson", s=18, zorder=5, label="R peaks")
    _overlay_events_on_axes(ax0, sid)

    # for lower panels: only draw event lines (no labels)
    _overlay_events_lines(ax1, sid)
    _overlay_events_lines(ax2, sid)

    # Fix x-limits to the selected window
    ax0.set_xlim(t0, t1)
    ax0.set_ylabel(ylabel)
    ax0.legend(loc="upper right", frameon=True)
    ax0.set_title(f"{sid} ECG zoom {t0:.1f}–{t1:.1f}s  |  fs≈{fs:.1f} Hz, peaks={pk.size}")

    if rr_ms is not None and tm is not None:
        rr_ms_arr = np.asarray(rr_ms, dtype=float)

        # --- Middle panel: ECG-derived RR ---
        # line: light green, points: blue
        ax1.plot(tm, rr_ms_arr, color="#b2df8a", linewidth=1.2, alpha=0.9, label="ECG→RR (line)")
        ax1.scatter(tm, rr_ms_arr, s=14, color="#1f78b4", alpha=0.9, zorder=4, label="ECG→RR (points)")

        # QC points (red x)
        if qc_flags is not None and np.any(qc_flags):
            ax1.scatter(tm[qc_flags], rr_ms_arr[qc_flags], marker='x', s=36,
                        linewidths=1.0, color='crimson', alpha=0.95, zorder=5, label='QC flagged')

        # robust y-limits for ECG-derived RR
        q1, q99 = np.nanpercentile(rr_ms_arr, [5, 95])
        span = max(1.0, float(q99 - q1))
        ax1.set_ylim(q1 - 0.2 * span, q99 + 0.2 * span)
        ax1.set_ylabel("RR (ms) — ECG")
        ax1.legend(loc="upper right", frameon=True)

        # --- Bottom panel: device RR (H10 native RR) in the same window ---
        dev_loaded = load_signal_raw(SRC_DIR, sid, "rr")
        if not dev_loaded:
            ax2.text(0.5, 0.5, "no device RR", transform=ax2.transAxes,
                     ha="center", va="center", color="gray")
            ax2.set_ylim(0, 1)
            ax2.set_yticks([])
        else:
            df_dev_rr, _ = dev_loaded
            df_dev_rr = df_dev_rr[(df_dev_rr["time_s"] >= t0) & (df_dev_rr["time_s"] <= t1)].copy()
            df_dev_rr = df_dev_rr.dropna(subset=["time_s", "value"])
            if df_dev_rr.empty:
                ax2.text(0.5, 0.5, "no device RR in window", transform=ax2.transAxes,
                         ha="center", va="center", color="gray")
                ax2.set_ylim(0, 1)
                ax2.set_yticks([])
            else:
                tt = df_dev_rr["time_s"].to_numpy(dtype=float)
                vv = df_dev_rr["value"].to_numpy(dtype=float)
                o = np.argsort(tt)
                tt, vv = tt[o], vv[o]

                # line: pale pink, points: orange-yellow
                ax2.plot(tt, vv, color="#f7b6d2", linewidth=1.1, alpha=0.85, label="device RR (line)")
                ax2.scatter(tt, vv, s=14, color="#f28e2b", alpha=0.9, zorder=4, label="device RR (points)")

                # robust y-limits for device RR
                q1d, q99d = np.nanpercentile(vv, [5, 95])
                spand = max(1.0, float(q99d - q1d))
                ax2.set_ylim(q1d - 0.2 * spand, q99d + 0.2 * spand)
                ax2.set_ylabel("RR (ms) — device")
                ax2.legend(loc="upper right", frameon=True)

    else:
        ax1.text(0.5, 0.5, "too few peaks", transform=ax1.transAxes,
                 ha="center", va="center", color="gray")
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])

        ax2.text(0.5, 0.5, "device RR not available", transform=ax2.transAxes,
                 ha="center", va="center", color="gray")
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])

    # Fix x-limits to the selected window
    ax1.set_xlim(t0, t1)
    ax2.set_xlim(t0, t1)
    ax2.set_xlabel("time (s)")
    ax1.grid(True, alpha=0.2)
    ax2.grid(True, alpha=0.2)

    out = (OUT_ROOT / f"{sid}_ecg_zoom.png").resolve()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[save] ecg-zoom → {out}")

def make_windowing_ecg_plot(
    sid: str,
    event: Optional[str] = None,
    t_start: Optional[float] = None,
    span: float = 10.0,
) -> Optional[dict]:
    """
    依据“事件名”或显式起始时间，绘制短窗 ECG 诊断图。
    优先使用事件名匹配到的时间；若无事件或匹配失败则回退到 t_start。
    两者皆无则给出提示并跳过。

    返回一个信息字典（包含选用的 t_start 与来源），失败返回 None。
    """
    # 1) 解析 t0 来源
    chosen_t0 = None
    source = None
    matched_label = None

    if event:
        ev = load_events(SRC_NORM_DIR, sid)
        if ev is not None and not ev.empty:
            # 事件名严格匹配（不区分大小写，去两端空白）；找不到再做“包含”匹配
            lab = ev["label"].astype(str).str.strip()
            key = str(event).strip()
            idx_exact = lab.str.casefold() == key.casefold()
            if idx_exact.any():
                chosen_t0 = float(ev.loc[idx_exact, "time_s"].iloc[0])
                source = "event_exact"
                matched_label = ev.loc[idx_exact, "label"].iloc[0]
            else:
                idx_contains = lab.str.casefold().str.contains(key.casefold(), na=False)
                if idx_contains.any():
                    chosen_t0 = float(ev.loc[idx_contains, "time_s"].iloc[0])
                    source = "event_contains"
                    matched_label = ev.loc[idx_contains, "label"].iloc[0]

    if chosen_t0 is None and t_start is not None:
        try:
            chosen_t0 = float(t_start)
            source = "t_start"
        except Exception:
            pass

    if chosen_t0 is None:
        print(f"[info] {sid}: no valid event/t_start for ECG zoom → skip")
        return None

    # 2) 边界检查：若窗口超出可用 ECG 时间范围，做温和修正（或缩窗）
    loaded = load_signal_raw(SRC_NORM_DIR, sid, "ecg")
    if not loaded:
        print(f"[warn] {sid}: cannot load ECG for windowing")
        return None
    df_ecg, _ = loaded
    if df_ecg.empty:
        print(f"[warn] {sid}: empty ECG for windowing")
        return None

    t_min = float(df_ecg["time_s"].min())
    t_max = float(df_ecg["time_s"].max())
    span = float(span)

    # 若 span 过大，限制为数据总长
    max_span = max(0.0, t_max - t_min)
    if span > max_span and max_span > 0:
        print(f"[info] {sid}: span={span:.3f}s > available={max_span:.3f}s; shrink span")
        span = max_span

    # 将 t0 调整到可行区间 [t_min, t_max - span]
    upper_start = max(t_min, t_max - span)
    if chosen_t0 < t_min:
        chosen_t0 = t_min
    if chosen_t0 > upper_start:
        chosen_t0 = upper_start

    # 3) 调用已有短窗绘图
    _plot_ecg_zoom_with_peaks_range(sid, t_start=chosen_t0, span_s=span)

    return {
        "sid": sid,
        "t_start": chosen_t0,
        "span": span,
        "source": source,
        "matched_label": matched_label,
    }

# --- 原始 rr 图 ---
def _plot_raw_signal(sid: str, sig: str):
    """Render one raw-scale plot for a given signal (excluding ACC). Saves to OUT_ROOT.
    RR is drawn as scatter in ms; others as lines in original units. Events are overlaid.
    """
    src = SRC_DIR if sig == "rr" else SRC_NORM_DIR
    loaded = load_signal_raw(src, sid, sig)
    if not loaded:
        return
    df_raw, ylabel = loaded
    if df_raw.empty:
        print(f"[warn] {sid}: empty {sig} in raw plot")
        return

    # Light aggregation for readability (units preserved)
    if sig == "rr":
        df_plot = df_raw  # event series → keep as-is
        agg_hz = None
    elif sig == "resp":
        df_plot, agg_hz = _prep_resp_for_plot(df_raw)
    else:
        if sig in ("ppg", "hr"):
            agg_hz = 2.0
        elif sig == "ecg":
            agg_hz = 10.0  # ECG 可适当降采样
        else:
            agg_hz = 1.0
        df_plot = _agg_value_per_hz(df_raw, hz=agg_hz) if len(df_raw) > 5000 else df_raw

    # Optional crop to event window and robust y-limit
    ev = load_events(SRC_NORM_DIR, sid)
    if ev is not None and not ev.empty:
        t0 = float(np.nanmin(ev["time_s"])) - 5.0
        t1 = float(np.nanmax(ev["time_s"])) + 5.0
        df_plot = df_plot[(df_plot["time_s"] >= t0) & (df_plot["time_s"] <= t1)]

    # Light smoothing for readability (non-RR)
    if sig != "rr":
        df_plot = _smooth_value(df_plot, win=3)

    # Robust y-limits from percentiles to avoid extreme outliers (units preserved)
    if df_plot["value"].notna().sum() >= 5:
        q1, q99 = np.nanpercentile(df_plot["value"], [1, 99])
        span = max(1e-9, q99 - q1)
        ymin, ymax = q1 - 0.1 * span, q99 + 0.1 * span
    else:
        ymin, ymax = None, None

    # Optional hard cap for RR raw plot to avoid end-of-session outliers dominating
    if sig == "rr" and RR_RAW_FORCE_YLIM is not None:
        ymin, ymax = RR_RAW_FORCE_YLIM

    plt.figure(figsize=(14, 4))
    ax = plt.gca()
    if sig == "rr":
        ax.scatter(df_plot["time_s"].values, df_plot["value"].values,
                   s=1, color="#f28e2b", alpha=0.9, label="RR", zorder=3)
    else:
        ax.plot(df_plot["time_s"].values, df_plot["value"].values,
                linewidth=1.0, alpha=0.95, color="#4E79A7", label=sig.upper(), zorder=2)

    # Mark RESP peaks with small red 'x' for quick visual counting
    if sig == "resp" and df_plot.shape[0] > 5:
        y = df_plot["value"].to_numpy(dtype=float)
        # robust prominence from 1–99% span
        q1, q99 = np.nanpercentile(y, [1, 99])
        prom = max(1e-6, 0.15 * (q99 - q1))
        # minimal distance between peaks derived from max breathing rate (≈30 bpm => period ≥2 s)
        resp_max_bpm = RESP_MAX_BPM
        alpha = 0.9  # keep slight margin to avoid merging true peaks
        min_dist = int(np.ceil((agg_hz or 2.0) * (60.0 / resp_max_bpm) * alpha))
        # optional minimal width (~0.3 s) in samples when SciPy is available
        width_pts = int(max(1, np.round((agg_hz or 2.0) * 0.3)))
        idx = _find_peaks_robust(y, distance=min_dist, prominence=prom, width=width_pts)
        if idx.size:
            ax.scatter(df_plot["time_s"].values[idx], y[idx], marker='x', s=18,
                       linewidths=0.8, color='fuchsia', alpha=0.9, zorder=4, label=None)

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    # Events overlay after y-limits applied
    _overlay_events_on_axes(ax, sid)

    ax.set_title(f"{sid} {sig.upper()} (raw scale)", fontsize=14)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.2)

    out = (OUT_ROOT / f"{sid}_{sig}_raw.png").resolve()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[save] raw-{sig} → {out}")

# --- RR→HR 1 Hz plot for cross-subject comparability ---
def _plot_rr_hr1hz(sid: str):
    """Plot RR converted to 1 Hz heart rate (bpm) with events overlay.
    Reads RR from confirmed/, converts HR=60000/rr_ms, aggregates to 1 Hz median.
    """
    loaded = load_signal_raw(SRC_DIR, sid, "rr")
    if not loaded:
        return
    df_raw, _ = loaded  # columns: time_s, value (rr_ms)
    if df_raw.empty:
        print(f"[warn] {sid}: empty rr for hr1hz plot")
        return

    # Convert to HR (bpm); ignore nonpositive values safely
    rr_ms = pd.to_numeric(df_raw["value"], errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        hr = 60000.0 / rr_ms
    df_hr = pd.DataFrame({"time_s": df_raw["time_s"].astype(float), "value": hr})
    df_hr = df_hr.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "value"]) 

    # 1 Hz aggregation for readability
    df_hr_1hz = _agg_value_per_hz(df_hr, hz=1.0)
    df_hr_1hz = _smooth_value(df_hr_1hz, win=3)

    # Crop to event span if available
    ev = load_events(SRC_NORM_DIR, sid)
    if ev is not None and not ev.empty:
        t0 = float(np.nanmin(ev["time_s"])) - 5.0
        t1 = float(np.nanmax(ev["time_s"])) + 5.0
        df_hr_1hz = df_hr_1hz[(df_hr_1hz["time_s"] >= t0) & (df_hr_1hz["time_s"] <= t1)]

    # Robust y-limits or forced limits in bpm
    if df_hr_1hz["value"].notna().sum() >= 5:
        q1, q99 = np.nanpercentile(df_hr_1hz["value"], [1, 99])
        span = max(1e-9, q99 - q1)
        ymin, ymax = q1 - 0.1 * span, q99 + 0.1 * span
    else:
        ymin, ymax = None, None
    if HR1HZ_FORCE_YLIM is not None:
        ymin, ymax = HR1HZ_FORCE_YLIM

    # Plot
    plt.figure(figsize=(14, 4))
    ax = plt.gca()
    ax.plot(df_hr_1hz["time_s"].values, df_hr_1hz["value"].values,
            linewidth=1.2, alpha=0.95, color="#f28e2b", label="HR (1 Hz)", zorder=3)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    _overlay_events_on_axes(ax, sid)

    ax.set_title(f"{sid} HR from RR (1 Hz)", fontsize=14)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.2)

    out = (OUT_ROOT / f"{sid}_hr1hz.png").resolve()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[save] hr1hz → {out}")

# --- 计算呼吸频率 peak ---
def _compute_resp_peaks_for_rate(df_raw: pd.DataFrame, rate_hz: float = 5.0) -> tuple[np.ndarray, float]:
    """
    将 RESP 聚合到 rate_hz（推荐 5 Hz）后找峰，返回 (t_peaks, hz)。
    仅用“时序”，不关心幅度。
    """
    # 聚合+平滑（不改变单位）
    df = _agg_value_per_hz(df_raw, hz=rate_hz)
    df = _smooth_value(df, win=5)
    # 再做 ~1 s 的均值平滑，稳定峰位
    k = max(3, int(round(rate_hz * 1.0)))
    df["value"] = pd.Series(df["value"].values).rolling(window=k, center=True, min_periods=1).mean()

    y = df["value"].to_numpy(dtype=float)
    t = df["time_s"].to_numpy(dtype=float)
    if y.size < 5:
        return np.array([], dtype=float), rate_hz

    # 显著性与最小间距：由采样率与最大呼吸频率（30 bpm）推得
    q1, q99 = np.nanpercentile(y, [1, 99])
    prom = max(1e-6, 0.15 * (q99 - q1))
    resp_max_bpm = 30.0
    alpha = 0.9
    min_dist = int(np.ceil(rate_hz * (60.0 / resp_max_bpm) * alpha))
    width_pts = int(max(1, round(rate_hz * 0.3)))  # ≥0.3 s

    idx = _find_peaks_robust(y, distance=min_dist, prominence=prom, width=width_pts)
    return (t[idx] if idx.size else np.array([], dtype=float)), rate_hz

# --- Helper: compute BR (breaths/min) time series from RESP peaks ---
def _compute_resp_br_series(df_raw: pd.DataFrame, rate_hz: float = 5.0) -> Optional[pd.DataFrame]:
    """Compute instantaneous breathing rate (BR, breaths/min) as a time series.
    Steps: (1) downsample to `rate_hz` and smooth; (2) detect peaks with robust params;
    (3) convert peak-to-peak intervals IBI to BR=60/IBI at each interval midpoint; (4) light median smoothing.
    Returns a DataFrame with columns: time_s, br_bpm. Returns None if peaks are insufficient.
    """
    t_peaks, hz = _compute_resp_peaks_for_rate(df_raw, rate_hz=rate_hz)
    if t_peaks.size < 3:
        return None
    ibi = np.diff(t_peaks)
    t_mid = (t_peaks[1:] + t_peaks[:-1]) / 2.0
    br = 60.0 / np.maximum(ibi, 1e-6)
    br_s = pd.Series(br).rolling(window=3, center=True, min_periods=1).median().to_numpy()
    return pd.DataFrame({"time_s": t_mid.astype(float), "br_bpm": br_s.astype(float)})

# --- 绘制呼吸频率图 ---
def _plot_resp_interval_views(sid: str):
    """
    输出“只看峰距”的 RESP 图（两行）：上=呼吸栅格，下=瞬时 BR。
    数据来自 norm/resp；忽略幅度，仅用峰时刻。
    """
    loaded = load_signal_raw(SRC_NORM_DIR, sid, "resp")
    if not loaded:
        return
    df_raw, _ = loaded
    if df_raw is None or df_raw.empty:
        print(f"[warn] {sid}: empty RESP for interval view"); return

    # 事件时间窗裁剪（可读性）
    ev = load_events(SRC_NORM_DIR, sid)
    if ev is not None and not ev.empty:
        t0 = float(np.nanmin(ev["time_s"])) - 5.0
        t1 = float(np.nanmax(ev["time_s"])) + 5.0
        df_raw = df_raw[(df_raw["time_s"] >= t0) & (df_raw["time_s"] <= t1)]

    # 以 5 Hz 找峰，更稳
    t_peaks, hz = _compute_resp_peaks_for_rate(df_raw, rate_hz=5.0)
    if t_peaks.size < 3:
        print(f"[info] {sid}: too few RESP peaks for interval view"); return

    # 由峰距得到 IBI 与瞬时 BR
    ibi = np.diff(t_peaks)                     # seconds
    t_mid = (t_peaks[1:] + t_peaks[:-1]) / 2.  # time positions for BR
    br = 60.0 / np.maximum(ibi, 1e-6)          # breaths/min

    # 轻度稳健平滑（去毛刺，不改变趋势）
    br_s = pd.Series(br).rolling(window=3, center=True, min_periods=1).median().to_numpy()

    # 绘图
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2]})

    # 上：呼吸栅格（tick train）。只画时间位置，不画幅度
    ax0.vlines(t_peaks, 0.0, 1.0, color="indigo", linewidth=0.5, alpha=0.6)
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.set_ylabel("breaths")
    _overlay_events_on_axes(ax0, sid)

    # 下：瞬时 BR 曲线
    ax1.plot(t_mid, br_s, linewidth=1.2, alpha=0.95, color="#4E79A7", label="BR (inst.)", zorder=3)
    if BR_FORCE_YLIM is not None:
        ax1.set_ylim(*BR_FORCE_YLIM)
    _overlay_events_on_axes(ax1, sid)
    ax1.set_ylabel("BR (breaths/min)")
    ax1.set_xlabel("time (s)")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper right", frameon=True)

    fig.suptitle(f"{sid} RESP peak intervals & breathing rate", fontsize=14)
    out = (OUT_ROOT / f"{sid}_resp_interval.png").resolve()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[save] resp-interval → {out}")

def plot_quick_preview(sid):
    signals = list(PRE_SIG) if isinstance(PRE_SIG, (list, tuple)) else []

    # Separate rr / acc / others to control layering and styles
    has_rr = "rr" in signals
    has_acc = "acc" in signals
    others = [s for s in signals if s not in ("rr", "acc")]

    # Colors and z-order
    color_rr = "#f28e2b"       # orange for RR (top layer)
    color_acc = "#bbbbbb"       # light grey for ACC (bottom layer)
    cool_colors = ["#4E79A7", "#59A14F", "#76B7B2", "#86BCB6", "#1f77b4", "#2ca02c"]

    plt.figure(figsize=(14, 4))
    ax = plt.gca()

    # 1) ACC first (from norm/): aggregate to ~1 Hz, light grey, bottom layer
    if has_acc:
        df_acc = load_signal_norm(SRC_NORM_DIR, sid, "acc")
        if df_acc is None or df_acc.empty:
            print(f"[warn] {sid}: cannot load ACC from norm")
        else:
            df_acc = _agg_per_hz(df_acc, hz=1.0)
            df_acc = _smooth(df_acc, win=3)
            ax.plot(df_acc["time_s"].values, df_acc["norm"].values,
                    label="ACC", linewidth=0.8, alpha=0.9, color=color_acc, zorder=1)

    # 2) Other continuous signals (from norm/): aggregate to 1–2 Hz, cool tones, middle layer
    for i, sig in enumerate(others):
        if sig == "resp":
            # Replace RESP amplitude with breathing rate (BR, breaths/min) for clearer reading.
            loaded = load_signal_raw(SRC_NORM_DIR, sid, "resp")
            if not loaded:
                print(f"[warn] {sid}: cannot load RESP from norm for preview")
                continue
            df_raw, _ = loaded
            if df_raw is None or df_raw.empty:
                print(f"[warn] {sid}: empty RESP in norm for preview")
                continue

            df_br = _compute_resp_br_series(df_raw, rate_hz=5.0)
            if df_br is None or df_br.empty:
                # fallback to readable RESP amplitude if BR cannot be computed
                df_resp, _ = _prep_resp_for_plot(df_raw)
                df_resp = df_resp.copy()
                df_resp["norm"] = _robust_minmax(df_resp["value"]).astype(float)
                ax.plot(df_resp["time_s"].values, df_resp["norm"].values,
                        label="RESP", linewidth=0.8, alpha=0.9,
                        color=cool_colors[i % len(cool_colors)], zorder=2)
                continue

            # Normalize BR to 0..1 for overlay on the mixed plot
            df_plot = df_br.copy()
            df_plot["norm"] = _robust_minmax(df_plot["br_bpm"]).astype(float)
            ax.plot(df_plot["time_s"].values, df_plot["norm"].values,
                    label="BR (resp)", linewidth=0.6, alpha=0.8,
                    color=cool_colors[i % len(cool_colors)], zorder=2)
            continue  # RESP handled; proceed to next signal

        # default: load normalized, light aggregation & smoothing
        df = load_signal_norm(SRC_NORM_DIR, sid, sig)
        if df is None or df.empty:
            print(f"[warn] {sid}: cannot load {sig} from norm")
            continue
        hz = 2.0 if sig in ("ppg", "hr") else 1.0
        df = _agg_per_hz(df, hz=hz)
        df = _smooth(df, win=3)
        ax.plot(df["time_s"].values, df["norm"].values,
                label=sig.upper(), linewidth=0.5, alpha=0.5,
                color=cool_colors[i % len(cool_colors)], zorder=2)

    # 3) RR last (from confirmed/): aggregate HR proxy to 1 Hz, orange solid, top layer
    if has_rr:
        df_rr = load_signal_norm(SRC_DIR, sid, "rr")
        if df_rr is None or df_rr.empty:
            print(f"[warn] {sid}: cannot load RR from confirmed")
        else:
            df_rr = _agg_per_hz(df_rr, hz=1.0)
            df_rr = _smooth(df_rr, win=3)
            ax.plot(df_rr["time_s"].values, df_rr["norm"].values,
                    label="RR", linewidth=1.2, alpha=0.95, color=color_rr, zorder=4)

    # 4) Events (from norm/): vertical red dashed lines with labels
    _overlay_events_on_axes(ax, sid)

    # axes cosmetics
    ax.set_title(f"{sid} quick {PRE_SIG} preview (normalized & aggregated)", fontsize=14)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized amplitude")
    ax.legend(loc="upper right", ncol=1, frameon=True)
    ax.grid(True, alpha=0.2)

    out = (OUT_ROOT / f"{sid}_quickcheck.png").resolve()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[save] preview → {out}")

    # 5) Also export raw-scale single-signal plots (exclude ACC). RR from confirmed/, others from norm/
    for sig in signals:
        if sig == "acc":
            continue
        _plot_raw_signal(sid, sig)

    # 6) Additionally, export RR→HR 1 Hz plot for cross-subject comparability
    if has_rr:
        _plot_rr_hr1hz(sid)

    # 7) Export RESP interval-only view (raster + instantaneous BR)
    if "resp" in signals:
        _plot_resp_interval_views(sid)

    # 8) Export ECG raw plot as well
    if "ecg" in signals:
        # 绘制全部ecg图
        _plot_ecg_signal(sid, "ecg")
        # 绘制制定窗口的edg
        make_windowing_ecg_plot(sid, event=ECG_PlOT_START_EVENT, t_start=ECG_PLOT_START, span=ECG_PLOT_SPAN)
def main():
    # Iterate over all PRE_SID with a progress bar
    sids = PRE_SID if isinstance(PRE_SID, (list, tuple)) else []
    if not sids:
        print("[info] preview_sids is empty; nothing to draw.")
    else:
        try:
            from tqdm import tqdm  # type: ignore
            iterable = tqdm(sids, desc="preview", unit="sid")
        except Exception:
            iterable = sids
        for _sid in iterable:
            plot_quick_preview(_sid)
if __name__ == "__main__":
    main()
    