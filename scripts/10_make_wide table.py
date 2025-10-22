from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any
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


DS = DATASETS[ACTIVE_DATA]
paths: Dict[str, Any] = DS["paths"]

PHYSICO_FILE = (DATA_DIR / paths["physico"]).resolve() / "physico.csv"
PSYCHO_FILE  = (DATA_DIR / paths["psycho"]).resolve() / "psycho.csv"
OUT_ROOT     = (DATA_DIR / paths["final"]).resolve() 
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_ROOT / "wide_table.csv"

# 列名配置（来自 settings）
PHYSICO_COLUMNS: List[str] = DS.get("signal_features", []) or []
PSYCHO_COLUMNS:  List[str] = DS.get("psycho_indices", []) or []

# w_id → phase 的固定映射
WID_TO_PHASE = {1: "baseline", 2: "induction", 3: "intervention", 4: "recovery"}
PHASE_ORDER = ["baseline", "induction", "intervention", "recovery"]


def _extract_sid_num(x) -> int | None:
    """从 subject_id 提取数字键：'P001S001...' → 1；'001' → 1；否则尽力转 int。"""
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x)
    m = re.search(r"P(\d{3})", s)
    if m:
        return int(m.group(1))
    try:
        return int(float(s))
    except Exception:
        return None


def _to_merge_id(x) -> str:
    """
    统一生成用于合并的键：优先提取数值 ID（如 P001 → 1），否则用原始 subject_id 的字符串形式。
    始终返回字符串，避免 dtype 不一致造成的 merge 报错。
    """
    n = _extract_sid_num(x)
    if n is not None:
        return str(int(n))
    return str(x).strip()


# === Loader functions ===
def _load_physico(path: Path) -> pd.DataFrame:
    """
    读取 physico.csv：只取必要与配置列；可选列 task/group/meaning 若存在则读取，不存在则忽略。
    使用 callable usecols，避免缺列触发 pandas Usecols 报错。
    """
    must_have = {"subject_id", "w_id"}
    desired = set(PHYSICO_COLUMNS) | must_have | {"task", "group", "meaning"}
    return pd.read_csv(path, usecols=lambda c: c in desired)


def _load_psycho(path: Path) -> pd.DataFrame:
    """
    读取 psycho.csv 的全部列（长表或历史宽表均可）。
    后续由 _pivot_psycho 根据是否存在 time 自动展开。
    """
    return pd.read_csv(path)

def _pivot_physio_no_agg(phy: pd.DataFrame) -> pd.DataFrame:
    """
    无聚合透视（每被试每阶段一行），确保 physio 宽表对 _merge_id 唯一：
    - 用 _merge_id 作为 pivot 唯一索引（字符串）
    - 若存在 group/task，先验证每个 _merge_id 的组别唯一，再在 pivot 后映射回去
    - 保留原始 subject_id（取该 _merge_id 第一条出现的值）
    """
    if "subject_id" not in phy.columns or "w_id" not in phy.columns:
        raise ValueError("physico.csv 需要列 'subject_id' 与 'w_id'。")

    df = phy.copy()

    # 统一合并键（字符串）
    df["_merge_id"] = df["subject_id"].apply(_to_merge_id)

    # w_id → phase
    df["w_id"] = pd.to_numeric(df["w_id"], errors="coerce").astype("Int64")
    df["phase"] = df["w_id"].map({1: "baseline", 2: "induction", 3: "intervention", 4: "recovery"})

    # 可能存在的组别列
    group_col = "task" if "task" in df.columns else ("group" if "group" in df.columns else None)

    # —— 组别唯一性校验，并构建映射（_merge_id → group）
    group_map = None
    if group_col:
        grp = df.loc[df[group_col].notna(), ["_merge_id", group_col]].drop_duplicates()
        dup_counts = grp.groupby("_merge_id")[group_col].nunique(dropna=True)
        violators = dup_counts[dup_counts > 1].index.tolist()
        if len(violators) > 0:
            raise ValueError(f"发现同一被试属于多个组别，请检查数据：{violators}")
        group_map = grp.drop_duplicates("_merge_id").set_index("_merge_id")[group_col]

    # 原始 subject_id 的代表值（用于输出展示）
    sid_map = df.groupby("_merge_id")["subject_id"].first()

    # 数值列：优先 settings.signal_features；否则用所有数值列
    if PHYSICO_COLUMNS:
        value_cols = [c for c in PHYSICO_COLUMNS
                      if c not in {"subject_id", "w_id", "task", "group", "meaning"} and c in df.columns]
    else:
        value_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "w_id"]

    if not value_cols:
        raise ValueError("physico.csv 未检测到可透视的数值列（请检查 settings.signal_features 或文件列名）。")

    # —— 用 _merge_id 作为唯一索引做透视（若极少数键重复，取 first）
    pivoted = pd.pivot_table(
        df, index=["_merge_id"], columns="phase", values=value_cols,
        aggfunc="first", dropna=False
    )
    pivoted.columns = [f"{metric}_{phase}" for metric, phase in pivoted.columns]
    wide = pivoted.reset_index()

    # 把 subject_id 与 group 映射回去（作为列）
    wide["subject_id"] = wide["_merge_id"].map(sid_map)
    if group_map is not None:
        wide[group_col] = wide["_merge_id"].map(group_map)

    # 列顺序整理（subject_id 放最前，其次 group，再指标×阶段，最后 _merge_id 以便调试）
    phase_order = ["baseline", "induction", "intervention", "recovery"]
    phys_cols_sorted: list[str] = []
    metrics = sorted({c.split("_")[0] for c in wide.columns
                      if "_" in c and c.rsplit("_", 1)[-1] in phase_order})
    for m in metrics:
        for p in phase_order:
            col = f"{m}_{p}"
            if col in wide.columns:
                phys_cols_sorted.append(col)

    front = ["subject_id"] + ([group_col] if group_col else [])
    other = [c for c in wide.columns if c not in set(front + phys_cols_sorted + ["_merge_id"])]
    wide = wide.reindex(columns=front + phys_cols_sorted + other + ["_merge_id"])

    # 确保 _merge_id 唯一
    if wide["_merge_id"].duplicated().any():
        dups = wide.loc[wide["_merge_id"].duplicated(), "_merge_id"].tolist()
        raise ValueError(f"构建后的 physio 宽表 _merge_id 仍非唯一，重复：{dups}")

    return wide

def _pivot_psycho(psy: pd.DataFrame) -> pd.DataFrame:
    """
    心理侧展开策略（使用 _merge_id 作为键）：
    A) 若存在 time 列（长表）：除 subject_id/time 外的所有列都按 *_T{time} 展开；
    B) 若不存在 time，但存在宽表列名形如 t{n}_var：重命名为 var_T{n} 并对 _merge_id 去重合并；
    其余列（如 flow 这类一次性指标）直接保留为单列。
    输出包含：_merge_id 以及各 *_T{t} 列（不保留 subject_id）。
    """
    if "subject_id" not in psy.columns:
        raise ValueError("psycho.csv 需要列 'subject_id'。")

    df = psy.copy()
    df["_merge_id"] = df["subject_id"].apply(_to_merge_id)

    # 情况 A：长表（推荐）
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
        value_cols = [c for c in df.columns if c not in {"subject_id", "_merge_id", "time"}]
        if not value_cols:
            raise ValueError("psycho.csv 仅含 subject_id 与 time，缺少心理指标列。")
        out = pd.DataFrame({"_merge_id": df["_merge_id"].drop_duplicates().sort_values()})
        for v in value_cols:
            pv = df.pivot_table(index="_merge_id", columns="time", values=v, aggfunc="first").reset_index()
            new_cols = {t: f"{v}_T{int(t)}" for t in pv.columns if t != "_merge_id" and pd.notna(t)}
            pv = pv.rename(columns=new_cols)
            keep = ["_merge_id"] + list(new_cols.values())
            out = out.merge(pv[keep], on="_merge_id", how="outer")
        cols = ["_merge_id"] + [c for c in out.columns if c != "_merge_id"]
        return out.reindex(columns=cols)

    # 情况 B：宽表（t{n}_var 形式）
    wide = df[["_merge_id"]].drop_duplicates().reset_index(drop=True)
    rename_map = {}
    time_pat = re.compile(r"^t(\d+)_([A-Za-z0-9_]+)$")
    for c in df.columns:
        if c in {"subject_id", "_merge_id"}:
            continue
        m = time_pat.match(c)
        if m:
            t = int(m.group(1))
            base = m.group(2)
            rename_map[c] = f"{base}_T{t}"
    renamed = df.rename(columns=rename_map)
    for c in [col for col in renamed.columns if col not in {"subject_id", "_merge_id"}]:
        s = renamed.groupby("_merge_id", as_index=False)[c].first()
        wide = wide.merge(s, on="_merge_id", how="left")
    cols = ["_merge_id"] + [c for c in wide.columns if c != "_merge_id"]
    return wide.reindex(columns=cols)


def build_wide_table() -> pd.DataFrame:
    phy = _load_physico(PHYSICO_FILE)
    psy = _load_psycho(PSYCHO_FILE)

    phy_wide = _pivot_physio_no_agg(phy)
    psy_wide = _pivot_psycho(psy)

    # 一人一行合并（键：_merge_id，字符串类型，避免 dtype 不一致）
    wide = phy_wide.merge(psy_wide, on="_merge_id", how="left", validate="1:1").drop(columns=["_merge_id"])

    # 可选：若存在典型列名则计算 Δ（不做其他派生）
    # STAI：优先使用 T1/T2/T3；若存在 T0/T1/T3 也计算
    if {"stai_T1", "stai_T2"}.issubset(wide.columns):
        wide["d_stai_induction"] = wide["stai_T2"] - wide["stai_T1"]
    elif {"stai_T0", "stai_T1"}.issubset(wide.columns):
        wide["d_stai_induction"] = wide["stai_T1"] - wide["stai_T0"]

    if {"stai_T2", "stai_T3"}.issubset(wide.columns):
        wide["d_stai_intervention"] = wide["stai_T3"] - wide["stai_T2"]
    elif {"stai_T1", "stai_T3"}.issubset(wide.columns):
        wide["d_stai_intervention"] = wide["stai_T3"] - wide["stai_T1"]

    # HRV：按阶段名
    if {"lnrmssd_baseline", "lnrmssd_induction"}.issubset(wide.columns):
        wide["d_lnrmssd_induction"] = wide["lnrmssd_induction"] - wide["lnrmssd_baseline"]
    if {"lnrmssd_induction", "lnrmssd_intervention"}.issubset(wide.columns):
        wide["d_lnrmssd_intervention"] = wide["lnrmssd_intervention"] - wide["lnrmssd_induction"]

    return wide


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wide = build_wide_table()
    wide.to_csv(OUT_FILE, index=False)
    print(f"[ok] 写出宽表：{OUT_FILE}")


if __name__ == "__main__":
    main()