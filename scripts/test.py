
from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "data").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent

file1 = "data/processed/windowing/local/collected/acc/P001S001T001R001_acc_w05.csv"
file2 = "data/processed/windowing/local/collected/acc/P001S001T001R001_acc_w06.csv"
file3 = "data/processed/windowing/local/collected/acc/P002S001T002R001_acc_w05.csv"
file4 = "data/processed/windowing/local/collected/acc/P002S001T002R001_acc_w06.csv"

# ---- Config (match your settings) ----
ACC_G_DEFAULT = 1000.0           # your settings['acc_g']
ACC_ENMO_THRESH_DEFAULT = 30.0   # your settings['acc_enmo_thresh']


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def find_time_col(df: pd.DataFrame) -> Optional[str]:
    # Common timestamp columns in this project / sensor exports
    candidates = [
        "t", "ts", "time", "timestamp", "lsl_ts", "lsl_time", "unix_ts", "host_ts",
        "t_s", "t_ms", "time_s", "time_ms",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Heuristic: a column that is strictly increasing and looks like time
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = _to_numeric_series(df[c]).dropna()
        if len(s) < 50:
            continue
        # allow some jitter, but should be mostly increasing
        inc_frac = float((np.diff(s.values) > 0).mean())
        if inc_frac > 0.98:
            return c
    return None


def find_acc_axis_cols(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    # Common axis naming conventions
    triplets = [
        ("ax", "ay", "az"),
        ("acc_x", "acc_y", "acc_z"),
        ("x", "y", "z"),
        ("X", "Y", "Z"),
        ("accX", "accY", "accZ"),
        ("ACC_X", "ACC_Y", "ACC_Z"),
    ]
    for a, b, c in triplets:
        if a in df.columns and b in df.columns and c in df.columns:
            return (a, b, c)

    # Heuristic fallback: choose the first 3 numeric columns excluding time-like
    tcol = find_time_col(df)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if tcol in numeric_cols:
        numeric_cols.remove(tcol)
    if len(numeric_cols) >= 3:
        return (numeric_cols[0], numeric_cols[1], numeric_cols[2])

    # More permissive: try to coerce columns, then pick those with many non-nan values
    cand = []
    for c in df.columns:
        s = _to_numeric_series(df[c])
        non_nan = int(s.notna().sum())
        if non_nan >= max(50, int(0.8 * len(df))):
            cand.append(c)
    if tcol in cand:
        cand.remove(tcol)
    if len(cand) >= 3:
        return (cand[0], cand[1], cand[2])

    return None


def estimate_unit_from_mag(mag_median: float) -> str:
    # Very rough but usually sufficient for diagnosing "why ENMO becomes 0"
    if 800 <= mag_median <= 1200:
        return "mG"           # Polar H10 acc is documented as mG
    if 0.8 <= mag_median <= 1.2:
        return "g"
    if 8.0 <= mag_median <= 12.0:
        return "m/s^2"
    return "unknown"


def enmo(vm: np.ndarray, acc_g: float) -> np.ndarray:
    return np.maximum(0.0, vm - acc_g)


def motion_frac(enmo_arr: np.ndarray, thresh: float) -> float:
    if len(enmo_arr) == 0:
        return float("nan")
    return float(np.mean(enmo_arr > thresh))


def describe_one_file(path: str) -> None:
    p = Path(path)
    print("=" * 88)
    print(f"FILE: {path}")
    if not p.exists():
        print("[ERROR] file not found")
        return

    # Read
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[ERROR] read_csv failed: {e}")
        return

    print(f"rows={len(df):,}  cols={len(df.columns)}")
    print("columns:", list(df.columns))

    tcol = find_time_col(df)
    axis = find_acc_axis_cols(df)

    if axis is None:
        print("[ERROR] Could not identify 3-axis acc columns. Please tell me your column names.")
        # show first few rows to help
        print(df.head(8).to_string(index=False))
        return

    axc, ayc, azc = axis
    print(f"time_col={tcol}  axis_cols={axis}")

    # Coerce
    ax = _to_numeric_series(df[axc]).to_numpy(dtype=float)
    ay = _to_numeric_series(df[ayc]).to_numpy(dtype=float)
    az = _to_numeric_series(df[azc]).to_numpy(dtype=float)

    # Drop rows where any axis is nan
    m = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
    ax, ay, az = ax[m], ay[m], az[m]
    if len(ax) == 0:
        print("[ERROR] No valid numeric rows after coercion.")
        return

    vm = np.sqrt(ax * ax + ay * ay + az * az)

    # Basic stats
    def q(x, qs=(0.05, 0.5, 0.95)):
        return np.quantile(x, qs)

    ax_q = q(ax)
    ay_q = q(ay)
    az_q = q(az)
    vm_q = q(vm)

    print("axis quantiles [p05, p50, p95]:")
    print(f"  {axc}: {ax_q[0]:.6g}, {ax_q[1]:.6g}, {ax_q[2]:.6g}")
    print(f"  {ayc}: {ay_q[0]:.6g}, {ay_q[1]:.6g}, {ay_q[2]:.6g}")
    print(f"  {azc}: {az_q[0]:.6g}, {az_q[1]:.6g}, {az_q[2]:.6g}")
    print("vm (vector magnitude) quantiles [p05, p50, p95]:")
    print(f"  vm:  {vm_q[0]:.6g}, {vm_q[1]:.6g}, {vm_q[2]:.6g}")

    vm_med = float(vm_q[1])
    unit_guess = estimate_unit_from_mag(vm_med)
    print(f"unit_guess_from_vm_median: {unit_guess} (vm_median={vm_med:.6g})")

    # Time step estimate if possible
    if tcol is not None and pd.api.types.is_numeric_dtype(df[tcol]):
        tt = _to_numeric_series(df.loc[m, tcol]).to_numpy(dtype=float)
        if len(tt) >= 5:
            d = np.diff(tt)
            d = d[np.isfinite(d)]
            if len(d) > 0:
                dt_med = float(np.median(d))
                dt_p95 = float(np.quantile(d, 0.95))
                # median absolute deviation / median for jitter
                mad = float(np.median(np.abs(d - np.median(d))))
                jitter = mad / dt_med if dt_med != 0 else float("inf")
                print(f"dt_median={dt_med:.6g}  dt_p95={dt_p95:.6g}  jitter(MAD/median)={jitter:.6g}")

    # Quantization / flatness checks
    def uniq_stats(x: np.ndarray, name: str) -> None:
        # Use rounding to avoid floating noise when data is integer-like
        xr = np.round(x, 6)
        n_unique = int(len(np.unique(xr)))
        print(f"{name}: n_unique(rounded6)={n_unique:,}  unique/len={n_unique/len(x):.6g}")

    uniq_stats(ax, axc)
    uniq_stats(ay, ayc)
    uniq_stats(az, azc)

    # ENMO diagnostics under different gravity assumptions
    # 1) User settings
    enmo_1000 = enmo(vm, ACC_G_DEFAULT)
    print("ENMO diagnostics:")
    print(f"  acc_g=1000 (settings): frac_zero={float(np.mean(enmo_1000==0)):.6g}  p95={float(np.quantile(enmo_1000,0.95)):.6g}")

    # 2) Alternative common units
    enmo_1 = enmo(vm, 1.0)
    enmo_98 = enmo(vm, 9.80665)
    print(f"  acc_g=1 (if unit is g): frac_zero={float(np.mean(enmo_1==0)):.6g}  p95={float(np.quantile(enmo_1,0.95)):.6g}")
    print(f"  acc_g=9.80665 (if unit is m/s^2): frac_zero={float(np.mean(enmo_98==0)):.6g}  p95={float(np.quantile(enmo_98,0.95)):.6g}")

    # 3) Session-adaptive gravity baseline
    g0 = float(np.median(vm))
    enmo_g0 = enmo(vm, g0)
    print(f"  acc_g=median(vm)={g0:.6g}: frac_zero={float(np.mean(enmo_g0==0)):.6g}  p95={float(np.quantile(enmo_g0,0.95)):.6g}")

    # Threshold-based motion fraction (convert threshold based on unit guess)
    if unit_guess == "mG":
        thr = ACC_ENMO_THRESH_DEFAULT
    elif unit_guess == "g":
        thr = ACC_ENMO_THRESH_DEFAULT / 1000.0
    elif unit_guess == "m/s^2":
        thr = (ACC_ENMO_THRESH_DEFAULT / 1000.0) * 9.80665
    else:
        thr = ACC_ENMO_THRESH_DEFAULT

    print(f"motion_frac using ENMO(acc_g={ACC_G_DEFAULT:.6g}) and thresh={thr:.6g} (unit_guess={unit_guess}): {motion_frac(enmo_1000, thr):.6g}")

    # Show a compact head for sanity
    head_cols: List[str] = []
    if tcol is not None:
        head_cols.append(tcol)
    head_cols += [axc, ayc, azc]
    print("head (selected cols):")
    print(df[head_cols].head(8).to_string(index=False))


def main() -> None:
    files = [file1, file2, file3, file4]
    for fp in files:
        describe_one_file(fp)


if __name__ == "__main__":
    main()