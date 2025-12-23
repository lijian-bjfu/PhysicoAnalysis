import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUT_ROOT = (SRC_DIR / "data_description" ).resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ！！因变量需要包含在 settings 中的 signal_features 列表中！！
dependent_vars = [
    "hf_log_ms2",
    "rmssd_ms",
    "mean_hr_bpm",
]
# 被测 ID 变量名,该变量为字符串，例如 "P006S001T002R001"
subject_id = "subject_id"
# 组间变量默认名
condition = "task"
# 时间变量默认名 t_id, 可切换为 phase_level_code
time = "t_id"
# 高亮特定的被试折线
# 颜色可选示例（matplotlib 基本颜色名称）：
# 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'black', 'cyan'
# 如需为更多被试定制样式，可在 SID_colormarker 中按 "PXXX" 添加条目，例如：
# "P012": {"color": "blue", "alpha": 0.8, "linewidth": 2.0}
SID_colormarker = {
    # cond 1
    # "P021": {
    #     "color": "red",
    #     "alpha": 0.9,
    #     "linewidth": 2.0,
    # },
    "P003": {
        "color": "green",
        "alpha": 0.9,
        "linewidth": 2.0,
    },
    "P017": {
        "color": "pink",
        "alpha": 0.9,
        "linewidth": 2.0,
    },
    "P001": {
        "color": "brown",
        "alpha": 0.9,
        "linewidth": 2.0,
    },
    "P007": {
        "color": "blue",
        "alpha": 0.9,
        "linewidth": 2.0,
    },
    # cond 2
    "P008": {
        "color": "orange",
        "alpha": 0.9,
        "linewidth": 2.0,
    },
    "P024": {
        "color": "pink",
        "alpha": 0.9,
        "linewidth": 2.0,
    },
}


# ------------------------------------------
# 数据描述与被试方差影响分析
# ------------------------------------------

def _describe_by_subject(df: pd.DataFrame, var: str, subject_col: str) -> pd.DataFrame:
    """
    对单个因变量，按被试汇总：均值/标准差/CV。
    """
    g = df.groupby(subject_col)[var]
    desc = g.agg(
        mean="mean",
        std="std",
        count="count",
    ).reset_index()
    # 对仅有 1 个观测值的被试，将 std 视为 0，便于后续计算 CV
    desc.loc[desc["count"] <= 1, "std"] = 0.0
    # 变异系数：标准差 / 均值；均值为 0 时设为 NaN
    desc["cv"] = desc["std"] / desc["mean"].replace({0: np.nan})
    return desc


def _describe_by_subject_time(df: pd.DataFrame, var: str, subject_col: str, time_col: str) -> pd.DataFrame:
    """
    对单个因变量，按 被试 × 时间窗口 汇总：均值/标准差/CV。
    """
    g = df.groupby([subject_col, time_col])[var]
    desc = g.agg(
        mean="mean",
        std="std",
        count="count",
    ).reset_index()
    # 对仅有 1 个观测值的 被试×时间 组合，将 std 视为 0
    desc.loc[desc["count"] <= 1, "std"] = 0.0
    desc["cv"] = desc["std"] / desc["mean"].replace({0: np.nan})
    return desc


def _subject_variance_influence(df: pd.DataFrame, var: str, subject_col: str) -> pd.DataFrame:
    """
    评估每个被试对总体方差（尤其是“被试间均值差异”部分）的贡献。

    使用经典的总平方和分解：
        Total SS = sum_{i,j} (y_ij - grand_mean)^2
        Between-subject SS = sum_i n_i * (mean_i - grand_mean)^2
        Within-subject SS = Total SS - Between-subject SS

    返回表中包含：
        - 每个被试的 n / mean / sd
        - 相对总体均值的偏差
        - 每个被试在 between-subject SS 中的贡献及其比例
    """
    # 仅保留当前变量和被试列
    data = df[[subject_col, var]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[var])
    if data.empty:
        return pd.DataFrame(columns=[
            subject_col, "n", "mean", "sd",
            "mean_diff_from_overall",
            "between_ss_contrib",
            "between_ss_contrib_pct_of_between",
            "between_ss_contrib_pct_of_total",
            "mean_z_across_subjects",
        ])

    grand_mean = data[var].mean()

    # 总平方和
    total_ss = ((data[var] - grand_mean) ** 2).sum()

    # 被试级均值与样本量
    g = data.groupby(subject_col)[var]
    subj_mean = g.mean()
    subj_count = g.count()
    subj_sd = g.std()

    # 被试间平方和：每个被试的贡献为 n_i * (mean_i - grand_mean)^2
    mean_diff = subj_mean - grand_mean
    between_ss_each = subj_count * (mean_diff ** 2)
    between_ss = between_ss_each.sum()

    # 被试内平方和：基于“每个观测相对本被试均值”的偏差
    tmp = data.join(subj_mean.rename("_subj_mean"), on=subject_col)
    within_ss = ((tmp[var] - tmp["_subj_mean"]) ** 2).sum()

    # 被试间均值的标准差，用于计算 Z 分数
    mean_sd_across_subjects = subj_mean.std(ddof=1)
    if mean_sd_across_subjects == 0 or np.isnan(mean_sd_across_subjects):
        mean_z = pd.Series(np.nan, index=subj_mean.index)
    else:
        mean_z = (subj_mean - subj_mean.mean()) / mean_sd_across_subjects

    res = pd.DataFrame({
        subject_col: subj_mean.index,
        "n": subj_count.values,
        "mean": subj_mean.values,
        "sd": subj_sd.values,
        "mean_diff_from_overall": mean_diff.values,
        "between_ss_contrib": between_ss_each.values,
        "mean_z_across_subjects": mean_z.values,
    })

    # 各被试在“被试间平方和”中的比例
    if between_ss > 0:
        res["between_ss_contrib_pct_of_between"] = res["between_ss_contrib"] / between_ss
    else:
        res["between_ss_contrib_pct_of_between"] = np.nan

    # 各被试在“总平方和”中的比例
    if total_ss > 0:
        res["between_ss_contrib_pct_of_total"] = res["between_ss_contrib"] / total_ss
    else:
        res["between_ss_contrib_pct_of_total"] = np.nan

    # 便于后续排序：贡献越大，说明该被试均值越“拖开”总体
    res = res.sort_values("between_ss_contrib_pct_of_total", ascending=False)
    return res.reset_index(drop=True)


def _subject_variance_influence_by_condition(df: pd.DataFrame, var: str, subject_col: str, cond_col: str) -> pd.DataFrame:
    """
    在每个条件组内分别评估被试对组内方差的贡献。
    """
    rows = []
    for cond_value, sub_df in df.groupby(cond_col):
        inf = _subject_variance_influence(sub_df, var, subject_col)
        if inf.empty:
            continue
        inf[cond_col] = cond_value
        rows.append(inf)
    if not rows:
        return pd.DataFrame(columns=[
            cond_col, subject_col, "n", "mean", "sd",
            "mean_diff_from_overall",
            "between_ss_contrib",
            "between_ss_contrib_pct_of_between",
            "between_ss_contrib_pct_of_total",
            "mean_z_across_subjects",
        ])
    res = pd.concat(rows, ignore_index=True)
    # 方便阅读：按条件、再按贡献排序
    res = res.sort_values([cond_col, "between_ss_contrib_pct_of_total"], ascending=[True, False])
    return res.reset_index(drop=True)


def _plot_subject_time_trajectories(df: pd.DataFrame, var: str, subject_col: str, time_col: str, cond_col: str, out_dir: Path) -> List[tuple]:
    """
    绘制被试在各时间点上的轨迹图，包括整体所有被试和各条件组内的被试轨迹及其均值。
    返回生成的图文件路径及描述列表。
    """
    fig_info: List[tuple] = []

    def _short_label(s):
        s_str = str(s)
        m = re.match(r"(P\d{3})", s_str)
        if m:
            return m.group(1)
        return s_str

    # 清理数据，保留需要列，去除缺失和无穷值
    plot_df = df[[subject_col, time_col, cond_col, var]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=[var, subject_col, time_col, cond_col])

    # 聚合：按条件、被试、时间，计算均值
    agg_df = plot_df.groupby([cond_col, subject_col, time_col])[var].mean().reset_index(name="mean")

    # 时间点排序
    unique_times = sorted(agg_df[time_col].unique())

    # 全体被试轨迹图
    fig, ax = plt.subplots(figsize=(12, 6))
    seen_labels = set()
    for subj, subdf in agg_df.groupby(subject_col):
        # 按时间排序
        subdf_sorted = subdf.set_index(time_col).reindex(unique_times).reset_index()
        label = _short_label(subj)

        # 根据被试 ID（例如 "P017"）查找是否有自定义样式
        # 优先使用缩写形式（PXXX），如果没有，再尝试完整 subject_id 作为键
        style_key = label
        style_cfg = SID_colormarker.get(style_key)
        if style_cfg is None:
            style_cfg = SID_colormarker.get(str(subj))

        # 默认线条样式
        line_kwargs = {
            "linewidth": 0.8,
            "alpha": 0.3,
            "marker": "o",
        }
        if style_cfg:
            if "color" in style_cfg:
                line_kwargs["color"] = style_cfg["color"]
            if "alpha" in style_cfg:
                line_kwargs["alpha"] = style_cfg["alpha"]
            if "linewidth" in style_cfg:
                line_kwargs["linewidth"] = style_cfg["linewidth"]

        if label in seen_labels:
            line_label = None
        else:
            line_label = label
            seen_labels.add(label)

        ax.plot(
            subdf_sorted[time_col],
            subdf_sorted["mean"],
            label=line_label,
            **line_kwargs,
        )
    # 计算整体均值（跨被试）
    mean_overall = agg_df.groupby(time_col)["mean"].mean().reindex(unique_times)
    ax.plot(unique_times, mean_overall,
            linewidth=2.0, alpha=0.9, marker="o", color="black", label="Overall mean")
    ax.set_xticks(unique_times)
    ax.set_xticklabels(unique_times, rotation=45)
    ax.set_xlabel(time_col)
    ax.set_ylabel(var)
    ax.set_title(f"{var} by {time_col}: subjects and overall mean")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    fig_path = out_dir / f"traj_{var}_by_{subject_col}_and_{time_col}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    fig_info.append((fig_path, f"{var} 被试轨迹图（含整体均值）"))

    # 各条件组内轨迹图
    for cond_value, cond_df in agg_df.groupby(cond_col):
        fig, ax = plt.subplots(figsize=(12, 6))
        seen_labels = set()
        for subj, subdf in cond_df.groupby(subject_col):
            subdf_sorted = subdf.set_index(time_col).reindex(unique_times).reset_index()
            label = _short_label(subj)

            # 根据被试 ID 应用自定义样式（若存在）
            style_key = label
            style_cfg = SID_colormarker.get(style_key)
            if style_cfg is None:
                style_cfg = SID_colormarker.get(str(subj))

            line_kwargs = {
                "linewidth": 0.8,
                "alpha": 0.3,
                "marker": "o",
            }
            if style_cfg:
                if "color" in style_cfg:
                    line_kwargs["color"] = style_cfg["color"]
                if "alpha" in style_cfg:
                    line_kwargs["alpha"] = style_cfg["alpha"]
                if "linewidth" in style_cfg:
                    line_kwargs["linewidth"] = style_cfg["linewidth"]

            if label in seen_labels:
                line_label = None
            else:
                line_label = label
                seen_labels.add(label)

            ax.plot(
                subdf_sorted[time_col],
                subdf_sorted["mean"],
                label=line_label,
                **line_kwargs,
            )
        mean_cond = cond_df.groupby(time_col)["mean"].mean().reindex(unique_times)
        ax.plot(unique_times, mean_cond,
                linewidth=2.0, alpha=0.9, marker="o", color="black", label="Condition mean")
        ax.set_xticks(unique_times)
        ax.set_xticklabels(unique_times, rotation=45)
        ax.set_xlabel(time_col)
        ax.set_ylabel(var)
        ax.set_title(f"{var} by {time_col} ({cond_col}={cond_value})")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        cond_sanitized = re.sub(r"[^0-9A-Za-z]+", "_", str(cond_value))
        fig_path = out_dir / f"traj_{var}_by_{subject_col}_and_{time_col}_cond_{cond_sanitized}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        fig_info.append((fig_path, f"{var} 被试轨迹图（{cond_col}={cond_value}）"))

    return fig_info


def _subject_interval_slopes_wide(
    df: pd.DataFrame,
    var: str,
    subject_col: str,
    time_col: str,
    cond_col: str,
) -> pd.DataFrame:
    """
    本函数对每个被试、每个相邻时间区间计算斜率（sl_*），组斜率（grp_*）以及差值（d_*），结果为宽表，每行一个被试。
    """
    data = df[[subject_col, cond_col, time_col, var]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[subject_col, cond_col, time_col, var])
    agg = data.groupby([cond_col, subject_col, time_col])[var].mean().reset_index(name="mean")

    unique_times = sorted(agg[time_col].unique())
    if len(unique_times) < 2:
        return pd.DataFrame()

    def _time_label(x):
        s = str(x)
        if s.endswith(".0"):
            return s[:-2]
        return s

    intervals = []
    for i in range(len(unique_times) - 1):
        t1 = unique_times[i]
        t2 = unique_times[i + 1]
        lab = f"{_time_label(t1)}_{_time_label(t2)}"
        intervals.append((t1, t2, lab))

    rows = []
    for cond_value, sub_df in agg.groupby(cond_col):
        pivot = sub_df.pivot(index=subject_col, columns=time_col, values="mean")
        pivot = pivot.reindex(columns=unique_times)

        slope_cols = {}
        for t1, t2, lab in intervals:
            dt = float(t2) - float(t1)
            if dt == 0:
                dt = 1.0
            slope = (pivot[t2] - pivot[t1]) / dt
            slope_cols[f"sl_{lab}"] = slope
        slopes_df = pd.DataFrame(slope_cols, index=pivot.index)

        group_slope = slopes_df.mean(axis=0, skipna=True)
        grp_df = pd.DataFrame(
            {f"grp_{col}": group_slope[col] for col in slopes_df.columns},
            index=slopes_df.index,
        )
        diff_df = slopes_df.sub(group_slope, axis=1)
        diff_df = diff_df.rename(columns={col: f"d_{col}" for col in diff_df.columns})

        out = pd.concat([slopes_df, grp_df, diff_df], axis=1)
        out.insert(0, cond_col, cond_value)
        out.index.name = subject_col
        out = out.reset_index()
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _round_numeric(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """
    将 DataFrame 中所有数值列按指定小数位数进行四舍五入，不修改非数值列。
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].round(digits)
    return df


def main():
    print(f"[info] 读取长表数据: {LONGTABLE}")
    df = pd.read_csv(LONGTABLE)

    # 检查必要列是否存在
    required_cols = {subject_id, condition, time}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"长表缺少必要列: {missing_required}")

    # 确保因变量存在
    global dependent_vars
    available_vars = [v for v in dependent_vars if v in df.columns]
    missing_vars = [v for v in dependent_vars if v not in df.columns]
    if missing_vars:
        print(f"[warn] 下列因变量未在长表中找到，将被跳过: {missing_vars}")
    if not available_vars:
        raise ValueError("在长表中未找到任何可用的因变量，请检查 dependent_vars 设置。")

    output_files = []

    for var in tqdm(available_vars, desc="variables"):
        print(f"[info] 处理因变量: {var}")

        # 1. 每个被试的描述统计
        by_subj = _describe_by_subject(df, var, subject_id)
        by_subj = _round_numeric(by_subj, digits=3)
        out1 = OUT_ROOT / f"desc_{var}_by_{subject_id}.csv"
        by_subj.to_csv(out1, index=False)
        output_files.append((out1, f"{var} 每被试总体描述（均值/标准差/CV）"))

        # 2. 每个被试 × 时间窗口 的描述统计
        by_subj_time = _describe_by_subject_time(df, var, subject_id, time)
        by_subj_time = _round_numeric(by_subj_time, digits=3)
        out2 = OUT_ROOT / f"desc_{var}_by_{subject_id}_and_{time}.csv"
        by_subj_time.to_csv(out2, index=False)
        output_files.append((out2, f"{var} 每被试 × 时间窗口 描述（均值/标准差/CV）"))

        # 3. 全局范围内，每个被试对方差的影响
        inf_overall = _subject_variance_influence(df, var, subject_id)
        inf_overall = _round_numeric(inf_overall, digits=3)
        out3 = OUT_ROOT / f"variance_influence_overall_{var}_by_{subject_id}.csv"
        inf_overall.to_csv(out3, index=False)
        output_files.append((out3, f"{var} 全局方差分解：各被试贡献"))

        # 4. 在每个条件组内的方差贡献
        inf_by_cond = _subject_variance_influence_by_condition(df, var, subject_id, condition)
        inf_by_cond = _round_numeric(inf_by_cond, digits=3)
        out4 = OUT_ROOT / f"variance_influence_by_{condition}_{var}_by_{subject_id}.csv"
        inf_by_cond.to_csv(out4, index=False)
        output_files.append((out4, f"{var} 分条件方差分解：各被试贡献"))

        # 5. 每个被试 × 相邻时间区间的斜率（宽表）
        slopes_wide = _subject_interval_slopes_wide(df, var, subject_id, time, condition)
        if not slopes_wide.empty:
            slopes_wide = _round_numeric(slopes_wide, digits=3)
            out5 = OUT_ROOT / f"slope_{var}_by_{subject_id}_intervals.csv"
            slopes_wide.to_csv(out5, index=False)
            output_files.append((out5, f"{var} 斜率描述（每被试 × 相邻时间区间，宽表）"))

        traj_figs = _plot_subject_time_trajectories(df, var, subject_id, time, condition, OUT_ROOT)
        output_files.extend(traj_figs)

    print("\n[info] 数据描述与方差贡献分析完成。生成的 CSV 文件如下：")
    for path, desc in output_files:
        print(f"  - {desc}: {path}")


if __name__ == "__main__":
    main()