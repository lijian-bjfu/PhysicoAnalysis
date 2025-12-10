import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import build_design_matrices
from scipy.stats import norm, chi2
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
OUT_ROOT = (SRC_DIR / "mixedlm_results" ).resolve()
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
# 排除的样本（按 subject_id 前缀过滤，例如以 P001 / P002 开头的被试）
exclude_SID = ["P001", "P002"]
# 固定因子,程序自动识别交叉
fixed_effects = [condition, time]
# 统计模型相关的全局参数
# 显著性水平
alpha: float = 0.05
# MixedLM 拟合方法：对应 SPSS /METHOD=ML
use_reml: bool = False  # False → ML, True → REML
# 最大迭代次数
max_iter: int = 100
# MixedLM 数值优化方法
optimize_method: str = "lbfgs"


def _filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """根据 exclude_SID 过滤被试，并删除关键变量缺失的观测。"""
    # 过滤被试前缀
    if exclude_SID:
        mask = ~df[subject_id].astype(str).str.startswith(tuple(exclude_SID))
        df = df.loc[mask].copy()

    # 仅保留分析中需要的列及非缺失行
    base_cols = [subject_id, condition, time]
    keep_cols = base_cols + [dv for dv in dependent_vars if dv in df.columns]
    df = df[keep_cols].copy()
    return df


def _fit_mixedlm(df: pd.DataFrame, dv: str):
    """
    针对单一因变量 dv 拟合混合线性模型：
        dv ~ C(task) * C(t_id)  + 随机截距(被试)
    """
    # 删除该因变量缺失的行
    cols = [subject_id, condition, time, dv]
    df_dv = df[cols].dropna().copy()

    # 将组别与时间显式视为类别变量
    df_dv[condition] = df_dv[condition].astype("category")
    df_dv[time] = df_dv[time].astype("category")

    formula = f"{dv} ~ C({condition}) * C({time})"

    model = smf.mixedlm(
        formula,
        data=df_dv,
        groups=df_dv[subject_id],  # 被试随机截距
    )
    result = model.fit(
        reml=use_reml,
        method=optimize_method,
        maxiter=max_iter,
    )
    return result, df_dv


def _extract_model_quality(result, dv: str) -> pd.DataFrame:
    """整理模型整体质量指标，格式类似 SPSS 的“模型拟合信息”表。"""
    try:
        n_groups = len(result.random_effects)
    except Exception:
        n_groups = np.nan

    # 随机截距方差与残差方差（注意：此处为独立残差结构，而非 SPSS 中的 AR(1) 残差结构）
    try:
        var_random_intercept = float(result.cov_re.iloc[0, 0])
    except Exception:
        var_random_intercept = np.nan

    try:
        converged_flag = bool(getattr(result, "converged", True))
    except Exception:
        converged_flag = True

    model_info = {
        "dv": dv,
        "n_obs": int(result.nobs),
        "n_groups": n_groups,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "loglik": float(result.llf),
        "converged": converged_flag,
        "var_random_intercept": var_random_intercept,
        "var_resid": float(result.scale),
    }
    return pd.DataFrame([model_info])


def _extract_fixed_effects(result, dv: str) -> pd.DataFrame:
    """提取固定效应结果表，包含估计值、标准误、z 值、p 值与置信区间。"""
    fe_names = result.fe_params.index
    beta = result.fe_params
    bse_all = result.bse
    bse_fe = bse_all[fe_names]
    z_vals = beta / bse_fe
    p_vals = 2 * (1 - norm.cdf(np.abs(z_vals)))
    z_arr = np.asarray(z_vals)
    p_arr = np.asarray(p_vals)

    ci_all = result.conf_int()
    ci_fe = ci_all.loc[fe_names]

    fixed_df = pd.DataFrame({
        "dv": dv,
        "term": fe_names,
        "estimate": beta.values,
        "std_error": bse_fe.values,
        "z": z_arr,
        "p": p_arr,
        "ci_low": ci_fe[0].values,
        "ci_high": ci_fe[1].values,
    })
    return fixed_df


def _type3_tests(result, dv: str) -> pd.DataFrame:
    """计算 Type III Wald 检验，针对 condition、time 及其交互作用的联合显著性检验。"""
    fe_names = list(result.fe_params.index)
    k = len(fe_names)
    rows = []

    beta = result.fe_params.loc[fe_names].values
    cov_fe = result.cov_params().loc[fe_names, fe_names].values

    # 主效应 condition
    idx_condition = [
        i for i, name in enumerate(fe_names)
        if re.match(f"^C\\({condition}\\)\\[", name) and ":" not in name
    ]
    if idx_condition:
        L = np.zeros((len(idx_condition), k))
        for row, col in enumerate(idx_condition):
            L[row, col] = 1.0
        L = np.asarray(L, dtype=float)
        theta = L @ beta  # shape (m,)
        V = L @ cov_fe @ L.T  # shape (m, m)
        try:
            stat = float(theta.T @ np.linalg.solve(V, theta))
        except np.linalg.LinAlgError:
            stat = float(theta.T @ np.linalg.pinv(V) @ theta)
        df_num = L.shape[0]
        p_val = float(1.0 - chi2.cdf(stat, df_num))
        rows.append({
            "dv": dv,
            "effect": condition,
            "num_df": df_num,
            "wald_chi2": stat,
            "p": p_val,
        })

    # 主效应 time
    idx_time = [i for i, name in enumerate(fe_names) if re.match(f"^C\\({time}\\)\\[", name)]
    if idx_time:
        L = np.zeros((len(idx_time), k))
        for row, col in enumerate(idx_time):
            L[row, col] = 1.0
        L = np.asarray(L, dtype=float)
        theta = L @ beta  # shape (m,)
        V = L @ cov_fe @ L.T  # shape (m, m)
        try:
            stat = float(theta.T @ np.linalg.solve(V, theta))
        except np.linalg.LinAlgError:
            stat = float(theta.T @ np.linalg.pinv(V) @ theta)
        df_num = L.shape[0]
        p_val = float(1.0 - chi2.cdf(stat, df_num))
        rows.append({
            "dv": dv,
            "effect": time,
            "num_df": df_num,
            "wald_chi2": stat,
            "p": p_val,
        })

    # 交互作用 condition:time
    idx_interaction = [i for i, name in enumerate(fe_names) if f"C({condition})" in name and f"C({time})" in name]
    if idx_interaction:
        L = np.zeros((len(idx_interaction), k))
        for row, col in enumerate(idx_interaction):
            L[row, col] = 1.0
        L = np.asarray(L, dtype=float)
        theta = L @ beta  # shape (m,)
        V = L @ cov_fe @ L.T  # shape (m, m)
        try:
            stat = float(theta.T @ np.linalg.solve(V, theta))
        except np.linalg.LinAlgError:
            stat = float(theta.T @ np.linalg.pinv(V) @ theta)
        df_num = L.shape[0]
        p_val = float(1.0 - chi2.cdf(stat, df_num))
        rows.append({
            "dv": dv,
            "effect": f"{condition}:{time}",
            "num_df": df_num,
            "wald_chi2": stat,
            "p": p_val,
        })

    return pd.DataFrame(rows)


def _emmeans_one_factor(result, df: pd.DataFrame, dv: str,
                        factor: str, other_factor: str) -> (pd.DataFrame, pd.DataFrame):
    """
    计算单一因素（task 或 t_id）的 estimated marginal means 及其两两比较结果，
    参照 SPSS /EMMEANS=TABLES(factor) COMPARE ADJ(BONFERRONI)。

    采用：在另一个因素的各水平上等权平均，再基于固定效应线性组合得到估计值与标准误。
    """
    design_info = result.model.data.orig_exog.design_info
    beta = result.fe_params
    cov_full = result.cov_params()
    cov_fe = cov_full.loc[beta.index, beta.index]

    levels_factor = list(df[factor].dropna().unique())
    levels_factor.sort()
    levels_other = list(df[other_factor].dropna().unique())
    levels_other.sort()

    rows = []
    L_map = {}

    # 计算每个水平的平均设计向量 L，并据此得到 emmean
    for lvl in levels_factor:
        grid_rows = []
        for lvl_other in levels_other:
            row = {
                factor: lvl,
                other_factor: lvl_other,
            }
            grid_rows.append(row)
        grid_df = pd.DataFrame(grid_rows)

        X = build_design_matrices([design_info], grid_df)[0]
        X_df = pd.DataFrame(X, columns=design_info.column_names)
        X_use = X_df[beta.index].to_numpy()

        L = X_use.mean(axis=0)
        L_map[lvl] = L

        emmean = float(L @ beta.values)
        var = float(L @ cov_fe.values @ L.T)
        se = float(np.sqrt(max(var, 0.0)))

        if se > 0:
            z_val = emmean / se
            p_val = 2 * (1 - norm.cdf(abs(z_val)))
        else:
            z_val = np.nan
            p_val = np.nan

        crit = norm.ppf(1 - alpha / 2)
        ci_low = emmean - crit * se
        ci_high = emmean + crit * se

        rows.append({
            factor: lvl,
            "dv": dv,
            "emmean": emmean,
            "se": se,
            "z": z_val,
            "p": p_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    emmeans_df = pd.DataFrame(rows)

    # 两两比较（Bonferroni 校正）
    pair_rows = []
    m = len(levels_factor)
    if m >= 2:
        n_comp = m * (m - 1) / 2
        for i in range(m):
            for j in range(i + 1, m):
                li = levels_factor[i]
                lj = levels_factor[j]
                Li = L_map[li]
                Lj = L_map[lj]
                Ldiff = Li - Lj

                diff = float(Ldiff @ beta.values)
                var_diff = float(Ldiff @ cov_fe.values @ Ldiff.T)
                se_diff = float(np.sqrt(max(var_diff, 0.0)))

                if se_diff > 0:
                    z_val = diff / se_diff
                    p_raw = 2 * (1 - norm.cdf(abs(z_val)))
                else:
                    z_val = np.nan
                    p_raw = np.nan

                # Bonferroni
                p_adj = min(p_raw * n_comp, 1.0) if p_raw is not None else np.nan

                pair_rows.append({
                    "dv": dv,
                    "factor": factor,
                    "level_1": li,
                    "level_2": lj,
                    "mean_diff": diff,
                    "se_diff": se_diff,
                    "z": z_val,
                    "p_raw": p_raw,
                    "p_bonferroni": p_adj,
                })

    pairs_df = pd.DataFrame(pair_rows)
    return emmeans_df, pairs_df


def _emmeans_cells(result, df: pd.DataFrame, dv: str,
                   factor_a: str, factor_b: str) -> pd.DataFrame:
    """
    计算二因素组合的 estimated marginal means，输出长表形式：
        factor_a, factor_b, emmean, se, ci_low, ci_high
    可分别用 (task, t_id) 与 (t_id, task) 调用，对应 SPSS 中的
    /EMMEANS=TABLES(task*t_id) 与 /EMMEANS=TABLES(t_id*task)。
    """
    design_info = result.model.data.orig_exog.design_info
    beta = result.fe_params
    cov_full = result.cov_params()
    cov_fe = cov_full.loc[beta.index, beta.index]

    levels_a = list(df[factor_a].dropna().unique())
    levels_a.sort()
    levels_b = list(df[factor_b].dropna().unique())
    levels_b.sort()

    rows = []

    for la in levels_a:
        for lb in levels_b:
            grid_df = pd.DataFrame([{factor_a: la, factor_b: lb}])
            X = build_design_matrices([design_info], grid_df)[0]
            X_df = pd.DataFrame(X, columns=design_info.column_names)
            X_use = X_df[beta.index].to_numpy()

            x_vec = X_use[0, :]
            emmean = float(x_vec @ beta.values)
            var = float(x_vec @ cov_fe.values @ x_vec.T)
            se = float(np.sqrt(max(var, 0.0)))

            if se > 0:
                z_val = emmean / se
                p_val = 2 * (1 - norm.cdf(abs(z_val)))
            else:
                z_val = np.nan
                p_val = np.nan

            crit = norm.ppf(1 - alpha / 2)
            ci_low = emmean - crit * se
            ci_high = emmean + crit * se

            rows.append({
                factor_a: la,
                factor_b: lb,
                "dv": dv,
                "emmean": emmean,
                "se": se,
                "z": z_val,
                "p": p_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

    emmeans_df = pd.DataFrame(rows)
    return emmeans_df


def run_mixedlm():
    """主入口：对 dependent_vars 中列出的所有因变量执行混合线性模型分析并输出结果表格。"""
    if not LONGTABLE.exists():
        raise FileNotFoundError(f"long_table.csv 不存在：{LONGTABLE}")

    df = pd.read_csv(LONGTABLE)
    df = _filter_data(df)

    all_model_info = []
    all_fixed = []
    all_emm_task = []
    all_emm_time = []
    all_emm_task_pairs = []
    all_emm_time_pairs = []
    all_emm_task_by_time = []
    all_emm_time_by_task = []
    all_type3 = []

    for dv in dependent_vars:
        if dv not in df.columns:
            print(f"[warn] 因变量 {dv} 不在 long_table 中，跳过。")
            continue

        print(f"[info] 拟合混合线性模型：dv = {dv}")
        result, df_dv = _fit_mixedlm(df, dv)

        model_info = _extract_model_quality(result, dv)
        fixed_df = _extract_fixed_effects(result, dv)
        type3_df = _type3_tests(result, dv)
        if not type3_df.empty:
            all_type3.append(type3_df)

        emm_task, pairs_task = _emmeans_one_factor(result, df_dv, dv, factor=condition, other_factor=time)
        emm_time, pairs_time = _emmeans_one_factor(result, df_dv, dv, factor=time, other_factor=condition)

        emm_task_time = _emmeans_cells(result, df_dv, dv, factor_a=condition, factor_b=time)
        emm_time_task = _emmeans_cells(result, df_dv, dv, factor_a=time, factor_b=condition)

        all_model_info.append(model_info)
        all_fixed.append(fixed_df)
        all_emm_task.append(emm_task)
        all_emm_time.append(emm_time)
        all_emm_task_pairs.append(pairs_task)
        all_emm_time_pairs.append(pairs_time)
        all_emm_task_by_time.append(emm_task_time)
        all_emm_time_by_task.append(emm_time_task)

    # 合并并输出各类结果表格
    if all_model_info:
        pd.concat(all_model_info, ignore_index=True).to_csv(
            OUT_ROOT / "mixedlm_model_fit.csv",
            index=False,
        )

    if all_fixed:
        pd.concat(all_fixed, ignore_index=True).to_csv(
            OUT_ROOT / "mixedlm_fixed_effects.csv",
            index=False,
        )

    if all_emm_task:
        pd.concat(all_emm_task, ignore_index=True).to_csv(
            OUT_ROOT / "mixedlm_emmeans_task.csv",
            index=False,
        )

    if all_emm_time:
        pd.concat(all_emm_time, ignore_index=True).to_csv(
            OUT_ROOT / "mixedlm_emmeans_time.csv",
            index=False,
        )

    if all_emm_task_pairs:
        pd.concat(all_emm_task_pairs, ignore_index=True).to_csv(
            OUT_ROOT / "mixedlm_emmeans_task_pairs.csv",
            index=False,
        )

    if all_emm_time_pairs:
        pd.concat(all_emm_time_pairs, ignore_index=True).to_csv(
            OUT_ROOT / "mixedlm_emmeans_time_pairs.csv",
            index=False,
        )

    if all_emm_task_by_time:
        pd.concat(all_emm_task_by_time, ignore_index=True).to_csv(
            OUT_ROOT / "mixedlm_emmeans_task_by_time.csv",
            index=False,
        )

    if all_emm_time_by_task:
        pd.concat(all_emm_time_by_task, ignore_index=True).to_csv(
            OUT_ROOT / "mixedlm_emmeans_time_by_task.csv",
            index=False,
        )

    if all_type3:
        pd.concat(all_type3, ignore_index=True).to_csv(
            OUT_ROOT / "mixedlm_type3_tests.csv",
            index=False,
        )


if __name__ == "__main__":
    run_mixedlm()