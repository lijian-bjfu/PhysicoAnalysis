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

# ----- 重要！------ #
# 最终宽表中的生理指标变量名字必须预先在settings.signal_features 注册，
# 否则无法出现在最终宽表中

# 列名配置（来自 settings）
PHYSICO_COLUMNS: List[str] = DS.get("signal_features", []) or []
PSYCHO_COLUMNS:  List[str] = DS.get("psycho_indices", []) or []

# w_id → phase 的固定映射
WID_TO_PHASE = {1: "baseline", 2: "induction", 3: "intervention", 4: "recovery"}

PHASE_ORDER = ["baseline", "induction", "intervention", "recovery"]
WINDOW_SETTING = DS["windowing"]

# === settings.windowing 解析辅助 ===
def _build_window_scheme() -> Dict[str, Any]:
    """
    从 settings.windowing（events_labeled_windows）构建统一解析方案：
      - 强制要求 use == 'events_labeled_windows'
      - 读取 window_category（如 ['p','g']）与 psycho_time（如 'g'）
      - 读取 windows 列表，并按出现顺序生成：
         * 对“phase 类别”（符号 != psycho_time）：列后缀 = name 的后缀文本（如 baseline/induction/intervention）
         * 对“gap 类别”（符号 == psycho_time）：列后缀 = t1/t2/t3（按在 settings 中出现的顺序编号）
      - 返回：
         {
           'use': 'events_labeled_windows',
           'psycho_sym': <str>,
           'cats': <List[str]>,
           'name_to_info': { name : {role, phase_level, t_level, col_suffix, t_id} },
           'suffix_orders': {'phase': [ ... ], 'gap': [ ... ]}
         }
    """
    ws = WINDOW_SETTING
    use = (ws or {}).get("use")
    if use != "events_labeled_windows":
        raise ValueError("本脚本仅支持 settings.windowing.use == 'events_labeled_windows'。若使用的是其它切窗模式，请改用 10_make_wide_table.py。")

    modes = (ws or {}).get("modes", {})
    mode_conf = (modes or {}).get(use, {})
    cats: List[str] = list(mode_conf.get("window_category", []))
    psycho_sym: str | None = mode_conf.get("psycho_time")
    wins: List[Dict[str, Any]] = list(mode_conf.get("windows", []))

    if not cats or not wins:
        raise ValueError("settings.windowing.modes.events_labeled_windows 缺少 window_category 或 windows 定义。")

    # 分组收集后缀（保持 settings 中出现顺序）
    group_suffixes: Dict[str, List[str]] = {sym: [] for sym in cats}
    valid_names: set[str] = set()
    for w in wins:
        name = str(w.get("name", "")).strip()
        if not name or "_" not in name:
            continue
        sym, suf = name.split("_", 1)
        if sym in group_suffixes:
            if suf not in group_suffixes[sym]:
                group_suffixes[sym].append(suf)
            valid_names.add(name)

    # 生成列后缀顺序
    phase_order: List[str] = []
    gap_order: List[str] = []
    name_to_info: Dict[str, Dict[str, Any]] = {}

    for sym, suf_list in group_suffixes.items():
        if not suf_list:
            continue
        if psycho_sym and sym == psycho_sym:
            # gap 维度：按出现顺序编号为 t1/t2/t3...
            gap_order = [f"t{i+1}" for i in range(len(suf_list))]
            # 为该类别建立 name→info
            for i, suf in enumerate(suf_list, start=1):
                # 对应窗口名
                name = f"{sym}_{suf}"
                name_to_info[name] = {
                    "role": "gap",
                    "phase_level": None,
                    "t_level": f"t{i}",
                    "col_suffix": f"t{i}",
                    "t_id": i
                }
        else:
            # phase 维度：后缀使用文本本身（baseline/induction/...）
            # 保持出现顺序
            for suf in suf_list:
                if suf not in phase_order:
                    phase_order.append(suf)
                name = f"{sym}_{suf}"
                name_to_info[name] = {
                    "role": "phase",
                    "phase_level": suf,
                    "t_level": None,
                    "col_suffix": suf,
                    "t_id": pd.NA
                }

    # 输出方案
    return {
        "use": use,
        "psycho_sym": psycho_sym,
        "cats": cats,
        "name_to_info": name_to_info,
        "suffix_orders": {
            "phase": phase_order,
            "gap": gap_order
        }
    }

SCHEME = _build_window_scheme()


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


# === meaning 清洗与窗口标签解析 ===
def _clean_meaning_name(val: str) -> str:
    """
    去掉 meaning 中的括号注释与首尾空白，只保留如 'p_baseline', 'g_t1', 'induction', 'gap_t2' 这类前缀。
    """
    if pd.isna(val):
        return ""
    s = str(val).strip()
    # 切掉类似 ' (内缩30.0s)' 的注释
    if " (" in s:
        s = s.split(" (", 1)[0].strip()
    return s

def _parse_window_label(meaning: str) -> Dict[str, Any]:
    """
    基于 settings.windowing.events_labeled_windows 的定义解析窗口标签：
      - 直接用 settings 中 windows 的 name 精确匹配（去除 meaning 中的括号注释后）
      - 属于 psycho_time 类别的窗口标记为 role='gap'，列后缀统一为 t1/t2/t3（按 settings 顺序）
      - 其余类别标记为 role='phase'，列后缀即文本后缀（baseline/induction/...）
    若未匹配到任何窗口名，返回空标记（后续 pivot 会忽略）。
    """
    s = _clean_meaning_name(meaning)
    info = SCHEME["name_to_info"].get(s)
    if info is None:
        # 未识别，返回空
        return {
            "role": "phase",
            "phase_level": None,
            "t_level": None,
            "col_suffix": "",
            "t_id": pd.NA
        }
    # 返回一个浅拷贝，避免外部修改
    return dict(info)


# === Loader functions ===
def _load_physico(path: Path) -> pd.DataFrame:
    """
    读取 physico.csv：只取必要与配置列；可选列 task/group/meaning 若存在则读取，不存在则忽略。
    使用 callable usecols，避免缺列触发 pandas Usecols 报错。
    备注：long 表构建需要 w_s/w_e，因此这里无论是否在 signal_features 中，都强制尝试读取；
         若文件中确无这两列，则在读取后补 NaN 占位，避免后续 keep_cols 选列时报 KeyError。
    """
    # 这些列是下游逻辑所需的“基础列”
    base_cols = {"subject_id", "w_id", "meaning", "task", "group", "w_s", "w_e"}
    # settings 中注册的生理指标列（含 *_ws / *_bs 等）
    desired = set(PHYSICO_COLUMNS) | base_cols

    phy = pd.read_csv(path, usecols=lambda c: c in desired)

    # 保底：若源文件缺失 w_s / w_e，则补出空列，避免后续选列时报错
    for col in ("w_s", "w_e"):
        if col not in phy.columns:
            phy[col] = np.nan

    return phy


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

    # 解析 meaning → role/phase_level/t_level/col_suffix/t_id
    df["meaning"] = df.get("meaning", "").astype(str)
    parsed = df["meaning"].apply(_parse_window_label)
    # 展开解析结果为多列
    df[["role", "phase_level", "t_level", "col_suffix", "t_id"]] = pd.DataFrame(parsed.tolist(), index=df.index)

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
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns) - {"w_id"}
    if PHYSICO_COLUMNS:
        # 将 settings 中的条目视作“基础名”（family），自动扩展其 _ws/_bs 派生列
        requested = [c for c in PHYSICO_COLUMNS if c not in {"subject_id", "w_id", "task", "group", "meaning"}]
        expanded: set[str] = set()
        for base in requested:
            # 1) 直接同名（显式列）
            if base in numeric_cols:
                expanded.add(base)
            # 2) 若 base 是“基础名”，则尝试附加 _ws/_bs
            for suf in ("_ws", "_bs"):
                cand = f"{base}{suf}"
                if cand in numeric_cols:
                    expanded.add(cand)
        # 3) 若 settings 已直接写入某个派生名（例如直接写了 xxx_ws），也别遗漏
        for c in requested:
            if c in numeric_cols:
                expanded.add(c)
        value_cols = sorted(expanded)
    else:
        value_cols = sorted(numeric_cols)

    if not value_cols:
        raise ValueError("physico.csv 未检测到可透视的数值列（请检查 settings.signal_features 或文件列名）。")

    # —— 用 _merge_id 作为唯一索引做透视（按 settings 顺序）
    suffix_order = SCHEME["suffix_orders"]["phase"] + SCHEME["suffix_orders"]["gap"]
    df_use = df[df["col_suffix"].isin(suffix_order)].copy()
    if df_use.empty:
        raise ValueError("未能从 meaning 解析出任何有效窗口。请检查 settings.windowing.modes.events_labeled_windows.windows 的 name 与 physico.csv 的 meaning 是否一致。")

    pivoted = pd.pivot_table(
        df_use, index=["_merge_id"], columns="col_suffix", values=value_cols,
        aggfunc="first", dropna=False
    )
    pivoted.columns = [f"{metric}_{suf}" for metric, suf in pivoted.columns]
    wide = pivoted.reset_index()

    # 把 subject_id 与 group 映射回去（作为列）
    wide["subject_id"] = wide["_merge_id"].map(sid_map)
    if group_map is not None:
        wide[group_col] = wide["_merge_id"].map(group_map)

    # 列顺序整理（subject_id 放最前，其次 group，再指标×阶段，最后 _merge_id 以便调试）
    phys_cols_sorted: list[str] = []
    metrics = sorted({c.rsplit("_", 1)[0] for c in wide.columns
                      if "_" in c and c.rsplit("_", 1)[-1] in suffix_order})
    for m in metrics:
        for suf in suffix_order:
            col = f"{m}_{suf}"
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


# === 生理长表构建与心理合并 ===
def _build_physio_long(phy: pd.DataFrame) -> pd.DataFrame:
    """
    基于解析后的窗口标签，构建适用于 LMM 的生理长表：
    列包含：subject_id, task/group(若有), role, phase_level, t_level, w_id, t_id, w_s, w_e, meaning, 以及各生理指标。
    其中：
      - phase 行：phase_level 有值，t_id 为空
      - gap 行：t_level 与 t_id 有值（t1=1,t2=2,t3=3）
    """
    if "subject_id" not in phy.columns or "w_id" not in phy.columns:
        raise ValueError("physico.csv 需要列 'subject_id' 与 'w_id'。")
    df = phy.copy()
    df["_merge_id"] = df["subject_id"].apply(_to_merge_id)

    # 解析 meaning
    df["meaning"] = df.get("meaning", "").astype(str)
    parsed = df["meaning"].apply(_parse_window_label)
    df[["role", "phase_level", "t_level", "col_suffix", "t_id"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    # 仅保留我们需要的数值列（与宽表一致）
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns) - {"w_id"}
    if PHYSICO_COLUMNS:
        requested = [c for c in PHYSICO_COLUMNS if c not in {"subject_id", "w_id", "task", "group", "meaning"}]
        expanded: set[str] = set()
        for base in requested:
            if base in numeric_cols:
                expanded.add(base)
            for suf in ("_ws", "_bs"):
                cand = f"{base}{suf}"
                if cand in numeric_cols:
                    expanded.add(cand)
        for c in requested:
            if c in numeric_cols:
                expanded.add(c)
        value_cols = sorted(expanded)
    else:
        value_cols = sorted(numeric_cols)

    keep_cols = ["subject_id", "_merge_id", "w_id", "t_id", "role", "phase_level", "t_level",
                 "w_s", "w_e", "meaning"] + value_cols

    # 附带 task/group
    group_col = "task" if "task" in df.columns else ("group" if "group" in df.columns else None)
    if group_col:
        keep_cols.insert(1, group_col)

    long_df = df[keep_cols].copy()
    # t_id 转 Int64
    if "t_id" in long_df.columns:
        long_df["t_id"] = pd.to_numeric(long_df["t_id"], errors="coerce").astype("Int64")
    return long_df

def _merge_psycho_into_long(long_df: pd.DataFrame, psy: pd.DataFrame) -> pd.DataFrame:
    """
    将心理长表（含 subject_id, time=1..3）按 t_id 合到生理长表：
    - 生理 phase 行（t_id 缺失）保持 NaN
    - gap 行（t_id=1/2/3）匹配 time=1/2/3 的心理指标
    """
    if "subject_id" not in psy.columns:
        return long_df

    dfp = psy.copy()
    dfp["_merge_id"] = dfp["subject_id"].apply(_to_merge_id)

    if "time" in dfp.columns:
        dfp["time"] = pd.to_numeric(dfp["time"], errors="coerce").astype("Int64")
        value_cols = [c for c in dfp.columns if c not in {"subject_id", "_merge_id", "time"}]
        # 先合并到 (_merge_id, time) 唯一
        dfp = dfp.groupby(["_merge_id", "time"], as_index=False)[value_cols].first()
        dfp = dfp.rename(columns={"time": "t_id"})
        merged = long_df.merge(dfp, on=["_merge_id", "t_id"], how="left")
        return merged

    # 没有 time 的话，直接按 _merge_id 合并（不建议）
    value_cols = [c for c in dfp.columns if c not in {"subject_id", "_merge_id"}]
    dfp = dfp.groupby(["_merge_id"], as_index=False)[value_cols].first()
    merged = long_df.merge(dfp, on=["_merge_id"], how="left")
    return merged

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

    # 宽表：生理（phase+gap 合表）
    phy_wide = _pivot_physio_no_agg(phy)

    # 心理宽表（维持原逻辑，得到如 stai_T1/T2/T3）
    psy_wide = _pivot_psycho(psy)

    # 合并宽表（以 _merge_id 为键）
    wide = phy_wide.merge(psy_wide, on="_merge_id", how="left", validate="1:1").drop(columns=["_merge_id"])

    # 可选 Δ 指标
    if {"stai_T1", "stai_T2"}.issubset(wide.columns):
        wide["d_stai_induction"] = wide["stai_T2"] - wide["stai_T1"]
    elif {"stai_T0", "stai_T1"}.issubset(wide.columns):
        wide["d_stai_induction"] = wide["stai_T1"] - wide["stai_T0"]

    if {"stai_T2", "stai_T3"}.issubset(wide.columns):
        wide["d_stai_intervention"] = wide["stai_T3"] - wide["stai_T2"]
    elif {"stai_T1", "stai_T3"}.issubset(wide.columns):
        wide["d_stai_intervention"] = wide["stai_T3"] - wide["stai_T1"]

    # 同时构建“长表”：生理 + 心理（心理只在 t_id 行有值）
    physio_long = _build_physio_long(phy)
    long_df = _merge_psycho_into_long(physio_long, psy)

    # 写长表
    long_path = OUT_ROOT / "long_table.csv"
    long_df.drop(columns=["_merge_id"], errors="ignore").to_csv(long_path, index=False)

    return wide


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wide = build_wide_table()
    wide.to_csv(OUT_FILE, index=False)
    print(f"[ok] 写出宽表：{OUT_FILE}")
    long_path = OUT_ROOT / "long_table.csv"
    print(f"[ok] 写出长表：{long_path}")
    print("[PREVIEW] 所有列名：")
    print(wide.columns)
    print("[PREVIEW] 最前 5 行：")
    print(wide.head())


if __name__ == "__main__":
    main()