import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Dict, Optional
from tqdm import tqdm

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATASETS, ACTIVE_DATA, DATA_DIR, SIGNAL_ALIASES, SCHEMA, PARAMS

# Dataset config
DS = DATASETS[ACTIVE_DATA]
paths = DS["paths"]
SRC_DIR = (DATA_DIR / paths["final"]).resolve()
LONGTABLE = (DATA_DIR / paths["final"] / "long_table.csv").resolve()
WIDETABLE = (DATA_DIR / paths["final"] / "wide_table.csv").resolve()
OUT_ROOT = (SRC_DIR / "data_explor" ).resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def _safe_name(s: str) -> str:
    """Make a string safe for use in filenames."""
    s = str(s)
    # keep letters/numbers/underscore/dash only
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")


def _tag_xyz(x: str, y: str, z: Optional[str]) -> str:
    """Build a compact tag that includes X/Y/Z names (Z may be None)."""
    z_tag = _safe_name(z) if z is not None else "noZ"
    return f"y_{_safe_name(y)}__x_{_safe_name(x)}__z_{z_tag}"

# 使用简写的被试ID
subject_id = "pid"

# 分组变量，0=实验组，1=对照组
condition = "task"

# 默认分析变量（可在此处快速切换）
# X: predictor
# Y: outcome
# Z: third variable for color/size encoding
X = "stai_T2"        
Y = "stai_T3"        
Z = "fss_selflessness_T3"


def scatter_by_group(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: Optional[str],
    group: str,
    out_path: Path,
    title: str,
):
    """
    Scatter plot of Y~X, split by group.
    Z is optionally used for color encoding.
    """
    groups = sorted(df[group].dropna().unique())

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        sub = df[df[group] == g].copy()

        # ensure numeric and drop missing values
        cols = [x, y] + ([z] if z is not None and z in sub.columns else [])
        for c in cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        sub = sub.dropna(subset=[x, y])

        if sub.empty:
            ax.set_title(f"{group} = {g} (no valid data)")
            continue

        if z is not None and z in sub.columns:
            sc = ax.scatter(
                sub[x],
                sub[y],
                c=sub[z],
                cmap="viridis",
                alpha=0.8,
                edgecolor="k",
            )
            plt.colorbar(sc, ax=ax, label=z)
        else:
            ax.scatter(sub[x], sub[y], alpha=0.8, edgecolor="k")

        # regression line (descriptive only, after NA handling)
        if sub.shape[0] >= 3:
            m, b = np.polyfit(sub[x], sub[y], 1)
            xs = np.linspace(sub[x].min(), sub[x].max(), 100)
            ax.plot(xs, m * xs + b)

        for _, r in sub.iterrows():
            ax.text(r[x], r[y], r[subject_id], fontsize=8, alpha=0.7)

        ax.set_title(f"{group} = {g}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def draw_hist(ax, fig, X, Y, Z, condition, OUT_ROOT):
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")

    fig.tight_layout()
    tag = _tag_xyz(X, Y, Z)
    fig.savefig(OUT_ROOT / f"hist__{tag}__by_{_safe_name(condition)}.png", dpi=300)
    plt.close(fig)


def main():
    print(f"[info] Loading data: {WIDETABLE}")
    df = pd.read_csv(WIDETABLE)

    required_cols = {subject_id, condition, X, Y}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("[info] Basic descriptive statistics")
    desc_cols = [c for c in [X, Y, Z, condition] if c in df.columns]
    print(df[desc_cols].apply(pd.to_numeric, errors="coerce").describe())

    # ---- Plot 1: Y ~ X by group, color = Z ----
    tag = _tag_xyz(X, Y, Z)
    out_file = OUT_ROOT / f"scatter__{tag}__by_{_safe_name(condition)}.png"
    scatter_by_group(
        df=df,
        x=X,
        y=Y,
        z=Z,
        group=condition,
        out_path=out_file,
        title=f"{Y} ~ {X} (colored by {Z})",
    )

    # ---- Plot 2: Y distribution by group ----
    fig, ax = plt.subplots(figsize=(6, 5))

    # Convert to numeric once for histogram use; keep original df intact.
    y_num = pd.to_numeric(df[Y], errors="coerce")

    # Use a shared binning so the two groups are directly comparable.
    y_all = y_num.dropna().to_numpy()
    if y_all.size >= 2:
        bin_edges = np.histogram_bin_edges(y_all, bins=10)
    else:
        bin_edges = 10

    for g in sorted(df[condition].dropna().unique()):
        mask = (df[condition] == g)
        sub = y_num[mask].dropna().to_numpy()
        if sub.size == 0:
            continue
        ax.hist(sub, bins=bin_edges, alpha=0.6, label=f"{condition}={g}")

    ax.set_xlabel(Y)
    ax.set_ylabel("Count")
    ax.legend()

    # Make x-axis tick labels readable (avoid the long repeating decimals problem).

    # draw_hist(ax, fig, X, Y, Z, condition, OUT_ROOT)

    print(f"[info] Figures saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()