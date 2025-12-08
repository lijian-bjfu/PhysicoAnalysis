from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import sys
from pathlib import Path

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATASETS, ACTIVE_DATA, DATA_DIR

# ---------------------------------------------------------------------
# 12_cal_Zscore
# 构建“Z分数变化”表用于快速做 t2-t1、t3-t2 的 z-change 分析（LMM或GLM）
#
# 读取：final/long_table.csv（只用 gap 角色下的 t_id=1,2,3）
# 变量：
#   - 生理（时域）：mean_hr_bpm -> hr；rmssd_ms -> rmssd
#   - RSA：rsa_ms -> rsa；rsa_log_ms -> rsa_log
#   - 心理：stai
#   - 保留的协变量：acc_enmo_mean_ws, resp_rate_bpm_ws, acc_enmo_mean_bs, resp_rate_bpm_bs
#
# 计算：
#   对每个指标 X，先在各自“组别”内计算 t1 的样本标准差 SD_g(t1) 与 t2 的样本标准差 SD_g(t2)（ddof=1）。
#   然后对每个被试：
#       X_12z = (X_t2 - X_t1) / SD_g(t1)   （诱导变化；分母使用该被试所属组别在 t1 的 SD）
#       X_23z = (X_t3 - X_t2) / SD_g(t2)   （恢复变化；分母使用该被试所属组别在 t2 的 SD）
#
# 输出长表（每被试 × 每指标 × 两个 z-change）：
#   subject_id : 被试
#   Zgroup     : 组别（=task，整数；1=涂抹组，2=点击组）
#   Ztime      : 变化段；1='t2-t1'(诱导变化)，2='t3-t2'(恢复变化)
#   Zindices   : 指标名（'hr','rmssd','rsa','rsa_log','stai'）
#   Zvalue     : 对应的 z-change 数值
#   协变量（便于后续 ANOVA/LMM 控制）：
#       acc_enmo_mean_ws, resp_rate_bpm_ws   （取“变化段的后一时点”的 ws 值：12z 用 t2，23z 用 t3）
#       acc_enmo_mean_bs, resp_rate_bpm_bs   （被试常量）
#
# 注：Ztime 的值标签（便于你在 SPSS 赋值标签）：
#   1 -> 't2 - t1'（诱导效应）
#   2 -> 't3 - t2'（恢复效应）
# ---------------------------------------------------------------------

DS = DATASETS[ACTIVE_DATA]
paths: Dict[str, Any] = DS["paths"]

LONG_TABLE = (DATA_DIR / paths["final"]).resolve() / "long_table.csv"
OUT_ROOT   = (DATA_DIR / paths["final"]).resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_ROOT / "zscore_table.csv"

# 想要参与 z-change 的原始列（来自 long_table）
INDEX_RAW_TO_LABEL: Dict[str, str] = {
    # 时域（简名）
    "mean_hr_bpm": "hr",
    "rmssd_ms": "rmssd",
    # RSA（两套）
    "rsa_ms": "rsa",
    "rsa_log_ms": "rsa_log",
    # 心理
    "stai": "stai",
}

# 将 Zindices 从字符串改为整数编码（便于 SPSS 作为分类因子使用）
# 1: hr, 2: rmssd, 3: rsa, 4: rsa_log, 5: stai
ZINDICES_CODE = {
    "hr": 1,
    "rmssd": 2,
    "rsa": 3,
    "rsa_log": 4,
    "stai": 5,
}
CODE_TO_ZINDICES = {v: k for k, v in ZINDICES_CODE.items()}

# 协变量列（ws 随时间、bs 为常量）
COV_WS = ["acc_enmo_mean_ws", "resp_rate_bpm_ws"]
COV_BS = ["acc_enmo_mean_bs", "resp_rate_bpm_bs"]

REQ_COLS = ["subject_id", "task", "role", "t_id"] + list(INDEX_RAW_TO_LABEL.keys()) + COV_WS + COV_BS


def _coerce_int(x) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _get_sd_by_group(gap: pd.DataFrame, col: str, t_id: int) -> Dict[int, float]:
    """
    在指定 t_id 上，按组别（task_int）分别计算“上一阶段”的样本标准差（ddof=1）。
    返回一个字典 {组别编码: SD}。若某组样本不足2个观测，返回 NaN。
    该实现遵循文献做法：变化(t_after - t_before) / SD(pre) 中的 SD 来自各自组别的 pre 时点。
    """
    out: Dict[int, float] = {}
    gi = gap.copy()
    if "task_int" not in gi.columns:
        gi["task_int"] = gi["task"].apply(_coerce_int)

    # 仅取目标 t_id 的数据，再按组别分别求 SD
    for g in sorted(set(gi["task_int"].dropna().unique())):
        s = pd.to_numeric(
            gi.loc[(gi["t_id"] == t_id) & (gi["task_int"] == g), col],
            errors="coerce"
        )
        s = s[np.isfinite(s)]
        out[int(g)] = float(np.std(s, ddof=1)) if s.size >= 2 else float("nan")
    return out


def _wide_by_tid(gap: pd.DataFrame, col: str) -> pd.DataFrame:
    """把某列按 (subject_id, task) × t_id 旋成 t1/t2/t3 宽表。"""
    sub = gap[["subject_id", "task", "t_id", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    pvt = sub.pivot_table(index=["subject_id", "task"], columns="t_id", values=col, aggfunc="first")
    for k in (1, 2, 3):
        if k not in pvt.columns:
            pvt[k] = np.nan
    pvt = pvt.rename(columns={1: "t1", 2: "t2", 3: "t3"})
    return pvt.reset_index()


def build_zscore_table() -> pd.DataFrame:
    # 读取长表
    df = pd.read_csv(LONG_TABLE)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] long_table 缺少必要列: {missing}")

    # 只保留 gap & t1/t2/t3
    gap = df.loc[(df["role"] == "gap") & (df["t_id"].isin([1, 2, 3]))].copy()

    # 类型清洗
    # task -> 尝试转 int；保留原值以防需要
    gap["task_int"] = gap["task"].apply(_coerce_int)
    # 确保 t_id 为 int
    gap["t_id"] = gap["t_id"].astype(int, errors="ignore")

    # 预先准备 BS 协变量（每被试常量，取首个非空）
    bs_map = (
        gap.groupby("subject_id")[COV_BS]
        .first()
        .reindex(columns=COV_BS)
        .to_dict(orient="index")
    )

    # 预先准备 WS 协变量在 t2/t3 的“查值”工具（便于 12z/23z 取后一时点）
    def _ws_at(sid: str, t_id: int, col: str) -> float:
        vals = gap.loc[(gap["subject_id"] == sid) & (gap["t_id"] == t_id), col]
        if not vals.empty:
            v = pd.to_numeric(vals.iloc[0], errors="coerce")
            return float(v) if np.isfinite(v) else float("nan")
        return float("nan")

    # 为 subject 级别的“宽格式 z 分数列”做累积容器（每个被试一行，包含 hr_12z、hr_23z...）
    subj_wide: Dict[str, Dict[str, Any]] = {}

    rows: List[Dict[str, Any]] = []

    # 遍历每个指标，分别算 12z、23z
    for raw_col, short_name in INDEX_RAW_TO_LABEL.items():
        # 计算分母 SD(t1/t2) 按组别
        sd_t1_by_g = _get_sd_by_group(gap, raw_col, 1)
        sd_t2_by_g = _get_sd_by_group(gap, raw_col, 2)

        wide = _wide_by_tid(gap, raw_col)

        for _, r in wide.iterrows():
            sid = r["subject_id"]
            task_int = r["task"]
            # 如果 task 不是纯数字，尝试解析
            if not isinstance(task_int, (int, np.integer)):
                task_int = _coerce_int(task_int)

            t1, t2, t3 = r.get("t1", np.nan), r.get("t2", np.nan), r.get("t3", np.nan)
            t1 = float(t1) if pd.notna(t1) else np.nan
            t2 = float(t2) if pd.notna(t2) else np.nan
            t3 = float(t3) if pd.notna(t3) else np.nan

            # 取该被试所属组别的 SD(pre)
            sd1 = sd_t1_by_g.get(int(task_int), float("nan")) if pd.notna(task_int) else float("nan")
            sd2 = sd_t2_by_g.get(int(task_int), float("nan")) if pd.notna(task_int) else float("nan")

            # z-change：12z（诱导段）
            z12 = np.nan
            if np.isfinite(t1) and np.isfinite(t2) and np.isfinite(sd1) and sd1 > 0:
                z12 = (t2 - t1) / sd1

            # z-change：23z（恢复段）
            z23 = np.nan
            if np.isfinite(t2) and np.isfinite(t3) and np.isfinite(sd2) and sd2 > 0:
                z23 = (t3 - t2) / sd2

            # —— 同时把本指标的 z 分数放入 subject 级别的“宽格式”字典 —— 
            if sid not in subj_wide:
                subj_wide[sid] = {}
            subj_wide[sid][f"{short_name}_12z"] = float(z12) if np.isfinite(z12) else np.nan
            subj_wide[sid][f"{short_name}_23z"] = float(z23) if np.isfinite(z23) else np.nan

            # 协变量（ws 用“后一时点”；bs 直接来自常量）
            bs = bs_map.get(sid, {})
            ws12 = {c: _ws_at(sid, 2, c) for c in COV_WS}  # 12z 用 t2
            ws23 = {c: _ws_at(sid, 3, c) for c in COV_WS}  # 23z 用 t3

            # 组织两行输出
            rows.append({
                "subject_id": sid,
                "Zgroup": task_int,       # 1=涂抹组, 2=点击填色组（源自 task）
                "Ztime": 1,               # 1='t2 - t1'（诱导变化）
                "Zindices": ZINDICES_CODE.get(short_name, np.nan),   # 指标编码
                "Zvalue": float(z12) if np.isfinite(z12) else np.nan,
                **{k: bs.get(k, np.nan) for k in COV_BS},
                **ws12,
            })
            rows.append({
                "subject_id": sid,
                "Zgroup": task_int,
                "Ztime": 2,               # 2='t3 - t2'（恢复变化）
                "Zindices": ZINDICES_CODE.get(short_name, np.nan),
                "Zvalue": float(z23) if np.isfinite(z23) else np.nan,
                **{k: bs.get(k, np.nan) for k in COV_BS},
                **ws23,
            })

    out = pd.DataFrame(rows)

    # 确保 Zindices 为整数编码（Int64 兼容缺失）
    if "Zindices" in out.columns:
        out["Zindices"] = pd.to_numeric(out["Zindices"], errors="coerce").astype("Int64")

    # ---------- 组装 subject 级“宽格式”表（每被试一行，包含 hr_12z、hr_23z...） ----------
    # 把 subj_wide 字典转为 DataFrame
    wide_df = pd.DataFrame.from_dict(subj_wide, orient="index").reset_index().rename(columns={"index": "subject_id"})
    # 添加被试级 BS 协变量（常量）
    for col in COV_BS:
        wide_df[col] = wide_df["subject_id"].map(lambda s: bs_map.get(s, {}).get(col, np.nan))
    # 为了与长表结构对齐，也保留 WS 协变量列，但 subject 层面无明确时点，置为 NaN
    for col in COV_WS:
        if col not in wide_df.columns:
            wide_df[col] = np.nan

    # ---------- 长表里把 subject_id 改名为 Zsubject_id，以便和 subject 级宽表并存 ----------
    out_long = out.copy()
    out_long = out_long.rename(columns={"subject_id": "Zsubject_id"})

    # 标记数据结构类型：1=wide(被试级汇总)，2=long(Z变化长表)
    wide_df["table_type"] = 1
    out_long["table_type"] = 2

    # ---------- 纵向拼接两种结构 ----------
    combined = pd.concat([wide_df, out_long], ignore_index=True, sort=False)

    # ---------- 最终列顺序 ----------
    z_cols = [
        "hr_12z", "hr_23z",
        "rmssd_12z", "rmssd_23z",
        "rsa_12z", "rsa_23z",
        "rsa_log_12z", "rsa_log_23z",
        "stai_12z", "stai_23z",
    ]
    final_cols = [
        "table_type",            # 1=wide, 2=long
        "subject_id",            # subject 级（宽表）所用
        "Zsubject_id", "Zgroup", "Ztime", "Zindices", "Zvalue",  # 长表（Z...）所用
    ] + z_cols + [
        "acc_enmo_mean_ws", "resp_rate_bpm_ws",  # WS（长表有值；subject 级宽表留空）
        "acc_enmo_mean_bs", "resp_rate_bpm_bs",  # BS（两边都能对齐）
    ]

    # 确保这些列都存在；没有的补 NaN
    for c in final_cols:
        if c not in combined.columns:
            combined[c] = np.nan

    # 只保留目标列并返回
    combined = combined[final_cols]

    # 简单排序：先按 subject_id，其次 Zsubject_id、Zindices、Ztime
    combined = combined.sort_values(by=["subject_id", "Zsubject_id", "Zindices", "Ztime"]).reset_index(drop=True)
    return combined


def main() -> None:
    zdf = build_zscore_table()
    zdf.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"[OK] 写出 Z 分数变化表: {OUT_FILE}")
    print("[PREVIEW] 前几行：")
    with pd.option_context("display.max_columns", 20, "display.width", 160):
        print(zdf.head(12))
    print("\n[提示] 值标签建议：Ztime -> {1:'t2 - t1', 2:'t3 - t2'}；Zgroup -> {1:'涂抹组', 2:'点击填色组'}")
    print("[提示] Zindices 为整数编码：1='hr', 2='rmssd', 3='rsa', 4='rsa_log', 5='stai'")

    # 打印一段可直接在 SPSS 中执行的值标签语法
    spss_syntax = (
        "VALUE LABELS table_type 1 'wide' 2 'long'.\n"
        "VALUE LABELS Ztime 1 't2 - t1' 2 't3 - t2'.\n"
        "VALUE LABELS Zgroup 1 '涂抹组' 2 '点击填色组'.\n"
        "VALUE LABELS Zindices 1 'hr' 2 'rmssd' 3 'rsa' 4 'rsa_log' 5 'stai'.\n"
        "FORMATS table_type Ztime Zgroup Zindices (F1.0).\n"
        "VARIABLE LABELS\n"
        " table_type '数据结构类型：1=wide 2=long'\n"
        " subject_id '被试ID（宽表行）'\n"
        " Zsubject_id '被试ID（Z长表行）'\n"
        " Zgroup '组别：1=涂抹 2=点击'\n"
        " Ztime '变化段：1=t2-t1 2=t3-t2'\n"
        " Zindices '指标编码：1=hr 2=rmssd 3=rsa 4=rsa_log 5=stai'\n"
        " Zvalue '标准化变化（z）'\n"
        " hr_12z 'HR z变化：t2-t1'\n"
        " hr_23z 'HR z变化：t3-t2'\n"
        " rmssd_12z 'RMSSD z变化：t2-t1'\n"
        " rmssd_23z 'RMSSD z变化：t3-t2'\n"
        " rsa_12z 'RSA(ms) z变化：t2-t1'\n"
        " rsa_23z 'RSA(ms) z变化：t3-t2'\n"
        " rsa_log_12z 'RSA(log) z变化：t2-t1'\n"
        " rsa_log_23z 'RSA(log) z变化：t3-t2'\n"
        " stai_12z 'STAI z变化：t2-t1'\n"
        " stai_23z 'STAI z变化：t3-t2'\n"
        " acc_enmo_mean_ws '加速度(WS,后一时点)'\n"
        " resp_rate_bpm_ws '呼吸频率(WS,后一时点)'\n"
        " acc_enmo_mean_bs '加速度(BS,被试常量)'\n"
        " resp_rate_bpm_bs '呼吸频率(BS,被试常量)'.\n"
    )
    print("\n[SPSS 值标签语法]\n" + spss_syntax)


if __name__ == "__main__":
    main()