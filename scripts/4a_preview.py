import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

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
OUT_ROOT = (DATA_DIR / paths["preview"] ).resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PRE_SID: List = DS["preview_sids"] # 选择 PRE_SID[0]
PRE_SIG: List = DS["windowing"]["apply_to"] # 选择列表中的所有信号
FILE_TYPES = ["csv", "parquet"]

# 不按照系统设定，临时检查文件
SRC_DIR = (DATA_DIR / paths["confirmed"])
PRE_SID = ["P003S001T001R001"]
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

# ---------------- plotting ----------------

def plot_quick_preview():
    if not isinstance(PRE_SID, (list, tuple)) or len(PRE_SID) == 0:
        print("[info] preview_sids is empty; nothing to draw.")
        return
    sid = PRE_SID[0]
    signals = list(PRE_SIG) if isinstance(PRE_SIG, (list, tuple)) else []

    # Prepare palette with high contrast
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(max(1, len(signals)))]

    plt.figure(figsize=(14, 4))
    ax = plt.gca()
    for i, sig in enumerate(signals):
        df = load_signal_norm(SRC_DIR, sid, sig)
        if df is None or df.empty:
            continue
        ax.plot(df["time_s"].values, df["norm"].values, label=sig.upper(), linewidth=1.0, alpha=0.9, color=colors[i])

    ax.set_title(f"{sid} quick {PRE_SIG} preview (normalized)", fontsize=14)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized amplitude")
    ax.legend(loc="upper right", ncol=1, frameon=True)
    ax.grid(True, alpha=0.2)

    out = (OUT_ROOT / f"{sid}_quickcheck.png").resolve()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[save] preview → {out}")

if __name__ == "__main__":
    plot_quick_preview()