from os import truncate
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
import difflib
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

def short_subject_id(s: str) -> str:
    """Convert full subject_id (e.g., 'P001S001T001R001') to a short form ('P001').

    If the expected pattern is not matched, the original string is returned.
    """
    s = str(s)
    m = re.match(r"^(P\d+)", s)
    return m.group(1) if m else s

# 个案 过滤与标注
subject_id = "subject_id"
# 使用简写形式在 plot 中标注个案，简写形式为 "P001"
use_short_subject_id = True 
show_subject_label = True
exclude = []
# 是否启用高亮（关闭时不高亮、也不显示高亮图注）
enable_highlight = True
# 突出显示某些个案
highlight_subject = ["P028","P009","P021","P017"]

# 分组变量，0=实验组，1=对照组
condition = "task"

# 默认分析变量（可在此处快速切换）
# X: predictor
# Y: outcome

X = "stai_T2"        
Y = "stai_T3"
# Z在不同模型下有不同含义。
# 在三变量图中表示: third variable for color/size encoding 
# 在交互图中表示控制变量
Z = ""

# 用户选择绘制哪些图
# draw_3vars_relathion_plot 打开后，绘制 scatter_by_group 的图
draw_3vars_relathion_plot = False
# draw_hist_plot 打开后，绘制直方图 draw_hist
draw_hist_plot = False
# draw_interaction_plot 打开后绘制交互图
draw_interaction_plot = True


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

    for gi, (ax, g) in enumerate(zip(axes, groups)):
        sub = df[df[group] == g].copy()

        # ensure numeric and drop missing values
        cols = [x, y] + ([z] if z is not None and z in sub.columns else [])
        for c in cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        sub = sub.dropna(subset=[x, y])

        if sub.empty:
            ax.set_title(f"{group} = {g} (no valid data)")
            continue

        # ---- Highlight specific subjects (optional) ----
        hl_set = {str(s) for s in highlight_subject} if (enable_highlight and highlight_subject) else set()
        sid_full = sub[subject_id].astype(str)
        sid_short = sid_full.map(short_subject_id)
        is_hl = sid_full.isin(hl_set) | sid_short.isin(hl_set)

        sub_norm = sub[~is_hl].copy()
        sub_hl = sub[is_hl].copy()

        if z is not None and z in sub.columns:
            # normal points use Z color encoding
            if not sub_norm.empty:
                sc = ax.scatter(
                    sub_norm[x],
                    sub_norm[y],
                    c=sub_norm[z],
                    cmap="viridis",
                    alpha=0.8,
                    edgecolor="k",
                )
                plt.colorbar(sc, ax=ax, label=z)

            # highlighted points use a fixed distinct style (marker+color)
            if not sub_hl.empty:
                hl_marker_cycle = ["*", "X", "P", "D", "^", "s"]
                hl_mk = hl_marker_cycle[gi % len(hl_marker_cycle)]
                ax.scatter(
                    sub_hl[x],
                    sub_hl[y],
                    marker=hl_mk,
                    s=160,
                    color="crimson",
                    alpha=0.95,
                    edgecolor="k",
                    linewidths=1.2,
                    label=(f"highlight ({group}={g})" if enable_highlight else None),
                )
        else:
            # no Z: draw normal points
            if not sub_norm.empty:
                ax.scatter(sub_norm[x], sub_norm[y], alpha=0.8, edgecolor="k")

            # highlighted points: fixed distinct style
            if not sub_hl.empty:
                hl_marker_cycle = ["*", "X", "P", "D", "^", "s"]
                hl_mk = hl_marker_cycle[gi % len(hl_marker_cycle)]
                ax.scatter(
                    sub_hl[x],
                    sub_hl[y],
                    marker=hl_mk,
                    s=160,
                    color="crimson",
                    alpha=0.95,
                    edgecolor="k",
                    linewidths=1.2,
                    label=(f"highlight ({group}={g})" if enable_highlight else None),
                )

        if show_subject_label:
            hl_set = {str(s) for s in highlight_subject} if (enable_highlight and highlight_subject) else set()
            for _, r in sub.iterrows():
                sid_full_r = str(r[subject_id])
                sid_short_r = short_subject_id(sid_full_r)
                is_hl_r = (sid_full_r in hl_set) or (sid_short_r in hl_set)

                lab = sid_short_r if use_short_subject_id else sid_full_r
                ax.text(
                    r[x],
                    r[y],
                    lab,
                    fontsize=9 if is_hl_r else 8,
                    alpha=0.95 if is_hl_r else 0.7,
                    fontweight="bold" if is_hl_r else "normal",
                    color="crimson" if is_hl_r else None,
                )
        if enable_highlight and (not sub_hl.empty):
            ax.legend(frameon=False)

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

def draw_interaction(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    out_path: Path,
    title: str,
    z: Optional[str] = None,
):
    """Plot the difference in the X→Y slope across groups (interaction-style visualization).

    This is a descriptive plot: it overlays per-group scatter points and per-group regression lines.
    """
    # Compact-but-readable figure size; legend is placed outside and saved with tight bounding box
    fig, ax = plt.subplots(figsize=(9, 6))

    groups = sorted(df[group].dropna().unique())
    marker_cycle = ["o", "s", "^", "D", "P", "X"]

    # Decide whether to control for Z
    use_z = (z is not None) and (str(z).strip() != "") and (str(z) in df.columns)

    # Prepare numeric columns once
    base_cols = [subject_id, group, x, y]
    if use_z:
        base_cols.append(str(z))

    work = df[base_cols].copy()
    work[x] = pd.to_numeric(work[x], errors="coerce")
    work[y] = pd.to_numeric(work[y], errors="coerce")
    if use_z:
        work[str(z)] = pd.to_numeric(work[str(z)], errors="coerce")
        work = work.dropna(subset=[x, y, group, str(z)])
    else:
        work = work.dropna(subset=[x, y, group])

    if work.empty:
        ax.set_title("No valid data")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    # Shared x-range for line drawing
    x_min, x_max = work[x].min(), work[x].max()
    xs = np.linspace(x_min, x_max, 200) if np.isfinite(x_min) and np.isfinite(x_max) else None

    for i, g in enumerate(groups):
        sub = work[work[group] == g].copy()
        if sub.empty:
            continue

        # If controlling Z, compute adjusted Y (holding Z at group mean)
        y_plot = y
        y_label = y
        if use_z:
            z_col = str(z)
            # Need at least 3 rows to estimate intercept + X + Z
            if sub.shape[0] >= 3:
                Xmat = np.column_stack([np.ones(len(sub)), sub[x].to_numpy(), sub[z_col].to_numpy()])
                beta, *_ = np.linalg.lstsq(Xmat, sub[y].to_numpy(), rcond=None)
                b0, b1, b2 = beta
                z_ref = float(np.nanmean(sub[z_col].to_numpy()))
                # Adjust Y to the reference Z so that points align with the controlled regression line
                sub["__y_adj__"] = sub[y] - b2 * (sub[z_col] - z_ref)
                y_plot = "__y_adj__"
                y_label = f"{y} (adj for {z_col})"
            else:
                # Not enough data to control; fall back to raw Y
                use_z = False

        mk = marker_cycle[i % len(marker_cycle)]

        # ---- Highlight specific subjects (optional) ----
        hl_set = {str(s) for s in highlight_subject} if (enable_highlight and highlight_subject) else set()
        sid_full = sub[subject_id].astype(str)
        sid_short = sid_full.map(short_subject_id)
        is_hl = sid_full.isin(hl_set) | sid_short.isin(hl_set)

        sub_norm = sub[~is_hl].copy()
        sub_hl = sub[is_hl].copy()

        if not sub_norm.empty:
            ax.scatter(
                sub_norm[x],
                sub_norm[y_plot],
                marker=mk,
                alpha=0.85,
                edgecolor="k",
                label=f"{group}={g}",
            )

        # highlighted points: fixed distinct style
        if not sub_hl.empty:
            hl_marker_cycle = ["*", "X", "P", "D", "^", "s"]
            hl_mk = hl_marker_cycle[i % len(hl_marker_cycle)]
            ax.scatter(
                sub_hl[x],
                sub_hl[y_plot],
                marker=hl_mk,
                s=160,
                color="crimson",
                alpha=0.95,
                edgecolor="k",
                linewidths=1.2,
                label=(f"{group}={g} highlight" if enable_highlight else None),
            )

        # regression line (descriptive only)
        if xs is not None:
            if use_z and ("__y_adj__" in sub.columns) and sub.shape[0] >= 3:
                # Use partial slope from Y ~ 1 + X + Z (plotted on adjusted Y)
                z_col = str(z)
                Xmat = np.column_stack([np.ones(len(sub)), sub[x].to_numpy(), sub[z_col].to_numpy()])
                beta, *_ = np.linalg.lstsq(Xmat, sub[y].to_numpy(), rcond=None)
                b0, b1, b2 = beta
                z_ref = float(np.nanmean(sub[z_col].to_numpy()))
                intercept_adj = b0 + b2 * z_ref
                ax.plot(xs, b1 * xs + intercept_adj, linewidth=2, label=f"{group}={g} fit")
            elif sub.shape[0] >= 2:
                m, b = np.polyfit(sub[x], sub[y_plot], 1)
                ax.plot(xs, m * xs + b, linewidth=2, label=f"{group}={g} fit")

        # optional subject labels
        if show_subject_label:
            hl_set = {str(s) for s in highlight_subject} if (enable_highlight and highlight_subject) else set()
            for _, r in sub.iterrows():
                sid_full_r = str(r[subject_id])
                sid_short_r = short_subject_id(sid_full_r)
                is_hl_r = (sid_full_r in hl_set) or (sid_short_r in hl_set)

                lab = sid_short_r if use_short_subject_id else sid_full_r
                ax.text(
                    r[x],
                    r[y_plot],
                    lab,
                    fontsize=9 if is_hl_r else 8,
                    alpha=0.95 if is_hl_r else 0.7,
                    fontweight="bold" if is_hl_r else "normal",
                    color="crimson" if is_hl_r else None,
                )

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y_label if 'y_label' in locals() else y)

    # Put legend on the right. Reserve figure space so it doesn't squeeze the axes.
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )

    # Use a tight bounding box so the exported image trims extra whitespace (including around the legend)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)



def main():
    print(f"[info] Loading data: {WIDETABLE}")
    df = pd.read_csv(WIDETABLE)

    # ---- Validate user-selected variable names (X/Y/Z/condition) with friendly messages ----
    cols = set(df.columns)

    # Required fields
    required_map = {
        "subject_id": subject_id,
        "group/condition": condition,
        "X": X,
        "Y": Y,
    }

    # Optional control/color variable
    z_is_defined = (Z is not None) and (str(Z).strip() != "")
    if z_is_defined:
        required_map["Z"] = Z

    missing_items = [(k, v) for k, v in required_map.items() if v not in cols]

    if missing_items:
        lines = []
        lines.append(f"[error] One or more variables were not found in the current data table: {WIDETABLE}")
        lines.append("[error] Please double-check the variable names (case-sensitive) against the column names in the table.")
        lines.append("")

        for role, varname in missing_items:
            # Suggest close matches from existing columns
            suggestions = difflib.get_close_matches(str(varname), df.columns.tolist(), n=5, cutoff=0.55)
            sug_txt = ("; did you mean: " + ", ".join(suggestions)) if suggestions else ""
            lines.append(f"  - Missing {role}: '{varname}'{sug_txt}")

        lines.append("")
        lines.append("[hint] You can open 'wide_table.csv' and copy/paste column names to avoid typos.")

        # Show a small sample of available columns to help orient the user.
        sample_cols = list(df.columns)[:30]
        more_txt = " ..." if len(df.columns) > 30 else ""
        lines.append(f"[hint] Available columns (first {len(sample_cols)}): {', '.join(sample_cols)}{more_txt}")

        raise ValueError("\n".join(lines))

    # ---- Optional: exclude specific subjects ----
    if exclude:
        # keep only ids that are NOT in exclude (support both full and short IDs)
        before_n = len(df)
        exclude_set = {str(x) for x in exclude}
        sid_full = df[subject_id].astype(str)
        sid_short = sid_full.map(short_subject_id)
        df = df[~(sid_full.isin(exclude_set) | sid_short.isin(exclude_set))].copy()
        after_n = len(df)
        print(f"[info] Excluded {before_n - after_n} rows by {subject_id} (full/short): {exclude}")

    print("[info] Basic descriptive statistics")
    desc_cols = [c for c in [X, Y, Z, condition] if c in df.columns]
    print(df[desc_cols].apply(pd.to_numeric, errors="coerce").describe())

    tag = _tag_xyz(X, Y, Z)

    # ---- Plot 1: Y ~ X by group, color = Z ----
    if draw_3vars_relathion_plot:
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
    if draw_hist_plot:
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
        draw_hist(ax, fig, X, Y, Z, condition, OUT_ROOT)

    # ---- Plot 3: Interaction-style slope difference plot (X→Y by group) ----
    if draw_interaction_plot:
        ctrl_tag = f"__ctrl_{_safe_name(Z)}" if (Z is not None and str(Z).strip() != "" and Z in df.columns) else ""
        out_file = OUT_ROOT / f"interaction__y_{_safe_name(Y)}__x_{_safe_name(X)}{ctrl_tag}__by_{_safe_name(condition)}.png"
        draw_interaction(
            df=df,
            x=X,
            y=Y,
            group=condition,
            out_path=out_file,
            title=f"Slope difference: {Y} ~ {X} by {condition}" + (f" (ctrl {Z})" if (Z is not None and str(Z).strip() != "" and Z in df.columns) else ""),
            z=Z,
        )

    print(f"[info] Output directory: {OUT_ROOT}")
    print(f"[info] draw_3vars_relathion_plot={draw_3vars_relathion_plot}, draw_hist_plot={draw_hist_plot}, draw_interaction_plot={draw_interaction_plot}")


if __name__ == "__main__":
    main()