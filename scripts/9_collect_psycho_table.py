import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import re
from collections import defaultdict

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

# 对原始量表的处理参数
# 只更新总指标（基于已生成的 psycho.csv 中的子量表列重新计算并覆盖）。例如 sms, fss。
only_cal_total_indices = True

# 需要反转的题项
reverse_items = ["Q1_Row1", "Q1_Row4", "Q1_Row5"]
# 反转题命名规则：原题目+"_r"
reverse_suffix = "_r"

# 题项构念。对原始量表进行 EFA / CFA 之后填写此变量
constructs = {
    # 状态焦虑
    "stai": ["Q1_Row1_r", "Q1_Row2", "Q1_Row3", "Q1_Row4_r", "Q1_Row5_r", "Q1_Row6"],
    # 反刍
    "rumi": ["Q2_Row1", "Q2_Row2", "Q2_Row5", "Q2_Row6", "Q2_Row11", "Q2_Row12", "Q2_Row17", "Q2_Row21"],
    # 正念-心智正念
    "smm": ["Q2_Row3", "Q2_Row4", "Q2_Row15", "Q2_Row16", "Q2_Row20", "Q2_Row10", "Q2_Row7"],
    # 正念-身体正念
    "sbm": ["Q2_Row8", "Q2_Row9", "Q2_Row13", "Q2_Row14", "Q2_Row18"],
    # 心流-技能挑战平衡
    "fss_balance": ["Q3_Row1", "Q3_Row8", "Q3_Row25", "Q3_Row9"],
    # 心流-注意与投入的自我控制
    "fss_con": ["Q3_Row4", "Q3_Row12", "Q3_Row20", "Q3_Row29", "Q3_Row13", "Q3_Row21", "Q3_Row30"],
    # 心流-不在乎他人
    "fss_self": ["Q3_Row5", "Q3_Row14", "Q3_Row22", "Q3_Row31"],
    # 心流-时间扭曲
    "fss_time": ["Q3_Row6", "Q3_Row15", "Q3_Row23"],
    # 心流-因任务而愉悦
    "fss_autotelic": ["Q3_Row7", "Q3_Row16", "Q3_Row24", "Q3_Row33"],
    # 易用性
    "peou": ["Q4_Row1", "Q4_Row2", "Q4_Row3", "Q4_Row4", "Q4_Row5"],
    # 努力
    "effort": ["Q5_Row1", "Q5_Row2", "Q5_Row3"],
    # 正面情绪
    "pa": ["Q6_Row1", "Q6_Row2", "Q6_Row3", "Q6_Row4"],
    # 人口学特征
    "demo": ['sex', 'age', 'art_exp', 'color_exp', 'art_join', 'ipad_exp', 'health'],
    # 将正念、心流汇总为总指标。当 only_cal_total_indices = True 时，仅更新这类变量
    "sms": ["smm", "sbm"],
    "fss": ["fss_balance", "fss_con", "fss_time", "fss_autotelic"],
    # 重复测量的各个时间点包含哪些量表。重复测量的指标以t为后缀，例如stai_t1, stai_t2, stai_t3。
    # 对于仅有一个时间点的数据不需要t后缀，如fss
    "t0": ["demo"],
    "t1": ["stai", "rumi", "smm", "sbm", "sms"],
    "t2": ["stai", "rumi", "smm", "sbm", "sms"],
    "t3": ["stai", "rumi", "smm", "sbm", "sms", "fss_balance", "fss_con", "fss_self", "fss_time", "fss_autotelic", "fss", "peou", "effort", "pa"],
}

# 分组变量
condition = "task"

def select_folder(title="请选择包含t0, t1, t2 CSV文件的文件夹"):
    """
    打开一个系统文件夹选择对话框，让用户选择文件夹。
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title="选择心理测量数据")
    root.destroy()
    
    if not folder_path:
        print("[信息] 用户取消了文件夹选择。程序退出。", file=sys.stderr)
        return None
        
    return Path(folder_path)


# ========== 评分函数与辅助 ==========
SCALE_MIN = 1
SCALE_MAX = 5

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric, leaving non-existing cols untouched."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def apply_reverse_items(df: pd.DataFrame, reverse_items: list[str], reverse_suffix: str) -> pd.DataFrame:
    """Create reversed item columns with suffix, using x_r = (min+max) - x."""
    if not reverse_items:
        return df
    df = df.copy()
    df = _coerce_numeric(df, reverse_items)
    for item in reverse_items:
        if item not in df.columns:
            # If the item is missing, create the reversed column as np.nan to make downstream strict scoring produce NA.
            df[item + reverse_suffix] = np.nan
            continue
        df[item + reverse_suffix] = (SCALE_MIN + SCALE_MAX) - df[item]
    return df

def _mean_strict_complete(df: pd.DataFrame, item_cols: list[str]) -> pd.Series:
    """Row-wise mean requiring complete data: if any item is NA, result is NA."""
    # Ensure columns exist; missing columns become NA
    work = pd.DataFrame(index=df.index)
    for c in item_cols:
        if c in df.columns:
            work[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            work[c] = np.nan
    # strict: any NA -> NA
    ok = work.notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index, dtype='float')
    out.loc[ok] = work.loc[ok].mean(axis=1)
    return out

def score_constructs_for_timepoint(
    df_raw: pd.DataFrame,
    time_point: str,
    constructs: dict,
    reverse_items: list[str],
    reverse_suffix: str,
    only_cal_total_indices: bool,
) -> pd.DataFrame:
    """Score constructs for one timepoint CSV.

    Returns a DataFrame indexed by subject_id containing:
      - time (int)
      - scored construct columns relevant to this time point
      - raw demo/static vars if present (for t0)
    """
    df = df_raw.copy()

    if 'subject_id' not in df.columns:
        raise KeyError(f"文件 '{time_point}.csv' 中缺失 'subject_id' 列。")

    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df = df.set_index('subject_id')

    # time as int
    m = re.search(r'\d+', time_point)
    time_num = int(m.group()) if m else 0

    # Reverse items (create *_r columns)
    df = apply_reverse_items(df, reverse_items, reverse_suffix)

    # Determine which construct names are requested at this time point
    requested = constructs.get(time_point, [])
    if not isinstance(requested, list):
        raise ValueError(f"constructs['{time_point}'] 必须为 list，当前为 {type(requested)}")

    # Resolve which columns to output when only_cal_total_indices=True.
    # Strategy: if a requested construct is a total (its definition references other constructs), we output only that total.
    # Any referenced subconstructs are computed internally but can be excluded from output.
    exclude_from_output = set()
    if only_cal_total_indices:
        # Totals are constructs whose definition list contains at least one token that is itself a construct key.
        for cname in requested:
            if cname not in constructs:
                continue
            deps = constructs[cname]
            if isinstance(deps, list) and any((d in constructs) for d in deps):
                for d in deps:
                    if d in constructs:
                        exclude_from_output.add(d)

    # Cache for computed constructs
    cache: dict[str, pd.Series] = {}

    def compute(name: str) -> pd.Series:
        if name in cache:
            return cache[name]

        if name not in constructs:
            # Not a defined construct; treat as raw column
            cache[name] = pd.to_numeric(df.get(name), errors='coerce') if name in df.columns else pd.Series(np.nan, index=df.index, dtype='float')
            return cache[name]

        items = constructs[name]
        if not isinstance(items, list) or len(items) == 0:
            cache[name] = pd.Series(np.nan, index=df.index, dtype='float')
            return cache[name]

        # If items reference other constructs, compute them first and take strict mean across those series.
        if any((it in constructs) for it in items):
            # build a temporary frame
            temp = pd.DataFrame(index=df.index)
            for it in items:
                if it in constructs:
                    temp[it] = compute(it)
                else:
                    # allow mixing raw column names too
                    temp[it] = pd.to_numeric(df.get(it), errors='coerce') if it in df.columns else np.nan
            ok = temp.notna().all(axis=1)
            out = pd.Series(np.nan, index=df.index, dtype='float')
            out.loc[ok] = temp.loc[ok].mean(axis=1)
            cache[name] = out
            return out

        # Otherwise treat as item columns
        cache[name] = _mean_strict_complete(df, items)
        return cache[name]

    # Always include time
    out = pd.DataFrame(index=df.index)
    out['time'] = time_num

    # For t0, allow demo/static raw vars passthrough
    # For other timepoints, we still allow passthrough if requested contains raw vars.
    for cname in requested:
        # compute if construct exists or raw column requested
        s = compute(cname)
        if only_cal_total_indices and cname in exclude_from_output:
            continue
        out[cname] = s

    return out


def _scan_timepoints(input_dir: Path) -> list[str]:
    """Scan folder for tN.csv files and return sorted timepoint names like ['t0','t1',...]."""
    hits = []
    for p in input_dir.glob('t*.csv'):
        m = re.match(r'^t(\d+)\.csv$', p.name)
        if m:
            hits.append((int(m.group(1)), f"t{int(m.group(1))}"))
    hits = sorted(hits, key=lambda x: x[0])
    return [h[1] for h in hits]


# ========== 仅基于 psycho.csv 更新总指标 ==========
def update_totals_in_existing_psycho(psycho_csv: Path) -> None:
    """Update total constructs in an existing psycho.csv.

    This mode is used when only_cal_total_indices=True.
    It recomputes totals defined in `constructs` that depend on other constructs
    (e.g., sms depends on smm/sbm; fss depends on fss_* subscales), using strict
    complete-case mean (any missing dependency -> NaN), and overwrites the total columns.
    """
    if not psycho_csv.exists():
        raise FileNotFoundError(f"未找到 psycho.csv: {psycho_csv}")

    df = pd.read_csv(psycho_csv)
    if df.empty:
        raise ValueError(f"psycho.csv 为空: {psycho_csv}")

    # Identify totals: constructs whose definition references other construct keys
    totals = []
    for k, v in constructs.items():
        if k in ("demo",) or re.match(r'^t\d+$', k):
            continue
        if isinstance(v, list) and len(v) > 0 and any((d in constructs) for d in v):
            totals.append(k)

    if not totals:
        print("[info] 未检测到需要更新的总指标（无 composite constructs）。")
        return

    def _row_mean_strict(cols: list[str]) -> pd.Series:
        temp = pd.DataFrame(index=df.index)
        for c in cols:
            if c in df.columns:
                temp[c] = pd.to_numeric(df[c], errors='coerce')
            else:
                temp[c] = np.nan
        ok = temp.notna().all(axis=1)
        out = pd.Series(np.nan, index=df.index, dtype='float')
        out.loc[ok] = temp.loc[ok].mean(axis=1)
        return out

    updated = []
    skipped = []
    for tname in totals:
        deps = constructs.get(tname, [])
        deps = [d for d in deps if isinstance(d, str)]
        if not deps:
            skipped.append((tname, "deps_empty"))
            continue
        # Compute strict mean across dependency columns
        df[tname] = _row_mean_strict(deps)
        updated.append((tname, deps))

    # Recompute post operations (deltas, centering, interactions) after totals update
    try:
        df_final, all_psycho_vars, delta_cols, centered_cols, inter_cols = apply_post_ops(df, constructs, condition)
    except Exception as e:
        print(f"[严重错误] 总指标已更新，但后操作重算失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Fallback: still write updated totals
        df_final = df

    df_final.to_csv(psycho_csv, index=False)

    if updated:
        for tname, deps in updated:
            print(f"[update] 已重算并覆盖总指标 '{tname}' <- mean_strict({deps})")
    if skipped:
        for tname, why in skipped:
            print(f"[skip] 总指标 '{tname}' 跳过：{why}")

    print(f"[done] psycho.csv 已更新（含后操作重算）：{psycho_csv}")

def process_data(input_dir: Path, out_root_path: Path, condi_file: Path | None = None):
    """
    读取CSV，评分构念，重塑为长表。
    """
    try:
        print("--- 1. 正在扫描时间点文件 (t0.csv, t1.csv, ...) ---")
        time_points = _scan_timepoints(input_dir)
        if not time_points:
            print("[致命错误] 未在所选文件夹中找到形如 t0.csv, t1.csv 的文件。", file=sys.stderr)
            return

        OUT_FILE = out_root_path / "psycho.csv"
        out_root_path.mkdir(parents=True, exist_ok=True)

        print(f"    将扫描的时间点文件: {time_points}")
        print(f"    only_cal_total_indices={only_cal_total_indices}")

        print("\n--- 2. 正在加载并评分各时间点CSV文件... ---")
        all_dfs = []

        for time_point in time_points:
            file_name = f"{time_point}.csv"
            file_path = input_dir / file_name
            if not file_path.exists():
                print(f"    [跳过] 文件不存在: {file_name}")
                continue

            print(f"    正在读取: {file_path}")
            df_raw = pd.read_csv(file_path)

            # 评分：按 constructs 定义生成构念列
            df_scored = score_constructs_for_timepoint(
                df_raw=df_raw,
                time_point=time_point,
                constructs=constructs,
                reverse_items=reverse_items,
                reverse_suffix=reverse_suffix,
                only_cal_total_indices=only_cal_total_indices,
            )

            all_dfs.append(df_scored)

        if not all_dfs:
            print("[致命错误] 未成功加载任何时间点CSV文件。", file=sys.stderr)
            return

        print("    CSV文件加载与评分完毕。")

        # --- 3. 合并 ---
        df_long = pd.concat(all_dfs, join='outer', ignore_index=False)
        df_long = df_long.reset_index().rename(columns={'index': 'subject_id'})

        # Derive static vars and repeated-measures vars from constructs configuration
        static_vars = list(constructs.get('demo', []))
        # Repeated-measures constructs: those present in at least 2 timepoints among t1..t9
        tp_keys = [k for k in constructs.keys() if re.match(r'^t\d+$', k)]
        tp_keys_sorted = sorted(tp_keys, key=lambda x: int(x[1:]))
        # collect construct names per timepoint (excluding 'demo' container)
        per_tp = {tp: [x for x in constructs.get(tp, []) if x != 'demo'] for tp in tp_keys_sorted}
        # candidates appear in >=2 timepoints
        freq = defaultdict(int)
        for tp, names in per_tp.items():
            for n in names:
                freq[n] += 1
        base_time_vars = sorted([n for n, c in freq.items() if c >= 2])

        # All psycho constructs that appear in any timepoint (including those that appear only once, e.g., t3-only scales)
        all_set = set()
        for tp in tp_keys_sorted:
            for n in per_tp.get(tp, []):
                if n == 'demo':
                    continue
                all_set.add(n)
        all_psycho_vars = sorted(all_set)

        print("    已合并为长表。正在填充静态变量...")

        # --- 静态变量填充 ---
        # 让同一个 subject_id 的所有行，都拥有这些静态值。
        for static_var in static_vars:
            if static_var in df_long.columns:
                df_long[static_var] = df_long.groupby('subject_id')[static_var].transform('first')

        # --- 从分组文件 (groups.csv) 合并条件变量，并编码为 0/1 ---
        cond01 = f"{condition}01"  # e.g., task01

        if condi_file is not None and Path(condi_file).exists():
            g = pd.read_csv(condi_file)
            if 'subject_id' not in g.columns or condition not in g.columns:
                raise KeyError(f"分组文件缺少必要列：需要 'subject_id' 与 '{condition}'。实际列为: {list(g.columns)}")

            # ！！！！ 注意：T1/T2/T3.csv 的 subject_id 与 groups.csv 的 subject_id 不同格式
            # - 问卷数据 subject_id：纯数字（1/2/3...，可能是字符串形式）
            # - groups.csv subject_id：形如 P001S001T001R001（真正的被试编号是 P 后面的数字 001 -> 1）
            # 因此：统一构造一个 numeric merge key（sid_num），在 sid_num 上对齐。

            # 1) 问卷长表侧：将 subject_id 转为数值 merge key
            df_long['_sid_num'] = pd.to_numeric(df_long['subject_id'], errors='coerce').astype('Int64')

            # 2) 分组文件侧：从 P 后提取数字作为 merge key
            g['subject_id'] = g['subject_id'].astype(str).str.strip()
            g['_sid_num'] = g['subject_id'].str.extract(r'^[Pp]\s*0*(\d+)', expand=False)
            g['_sid_num'] = pd.to_numeric(g['_sid_num'], errors='coerce').astype('Int64')

            # 3) task 列尽量转数值（允许原始为字符串 '1'/'2'）
            g[condition] = pd.to_numeric(g[condition], errors='coerce')

            # 4) 合并：在 _sid_num 上左连接（保持问卷长表的行）
            df_long = df_long.merge(g[['_sid_num', condition]], on='_sid_num', how='left')

            # 5) 合并诊断：如果完全未命中，提示可能的原因
            n_total = len(df_long)
            n_hit = int(df_long[condition].notna().sum())
            hit_rate = (n_hit / n_total) if n_total else 0.0
            print(f"    分组对齐: 命中 {n_hit}/{n_total} 行 (hit_rate={hit_rate:.3f})")
            if n_hit == 0:
                left_ids = sorted(df_long['_sid_num'].dropna().unique().tolist())[:10]
                right_ids = sorted(g['_sid_num'].dropna().unique().tolist())[:10]
                print(f"[警告] 分组对齐失败：问卷侧 sid_num 示例(前10)={left_ids}; groups侧 sid_num 示例(前10)={right_ids}。\n"
                      f"       请检查：问卷 subject_id 是否确为纯数字；groups.csv 是否以 P 开头且含数字编号；以及两边编号是否同一套。",
                      file=sys.stderr)

            # 6) 清理临时 merge key（保留原始 subject_id，不改变输出标识）
            df_long = df_long.drop(columns=['_sid_num'], errors='ignore')

            # 编码为 0/1：按数值从小到大映射
            s = df_long[condition]
            uniq = sorted(pd.unique(s.dropna()).tolist())
            if len(uniq) == 0:
                df_long[cond01] = pd.Series(pd.NA, index=df_long.index, dtype='Int64')
                print(f"[警告] 分组文件 '{condi_file}' 中未检测到有效的 '{condition}' 值，已创建全缺失列 '{cond01}'。", file=sys.stderr)
            elif set(uniq).issubset({0, 1}) and len(uniq) <= 2:
                df_long[cond01] = s.astype('Int64')
            elif len(uniq) == 2:
                mapping = {uniq[0]: 0, uniq[1]: 1}
                df_long[cond01] = s.map(mapping).astype('Int64')
                print(f"    条件变量 '{condition}' 数值编码映射: {mapping}")
            else:
                mapping = {uniq[0]: 0, uniq[1]: 1}
                df_long[cond01] = s.map(mapping).astype('Int64')
                print(f"[警告] 条件变量 '{condition}' 检测到 >2 个水平: {uniq}。仅保留前两类映射 {mapping}，其余置为缺失。", file=sys.stderr)

        else:
            # 没有提供分组文件或文件不存在
            df_long[cond01] = pd.Series(pd.NA, index=df_long.index, dtype='Int64')
            print(f"[警告] 分组文件不存在或未提供：{condi_file}。已创建全缺失列 '{cond01}'。", file=sys.stderr)

        # 输出只保留 task01，不保留原始 task
        if condition in df_long.columns:
            df_long = df_long.drop(columns=[condition])

        # --- 3) 后操作（差值、去中心化、交互项） ---
        df_final, all_psycho_vars, delta_cols, centered_cols, inter_cols = apply_post_ops(df_long, constructs, condition)

        # --- 4. 保存 ---
        df_final.to_csv(OUT_FILE, index=False)

        print("\n--- 3. 处理完成 ---")
        print(f"    最终长表保存至: {OUT_FILE}")
        print(df_final.head(10))  # 打印多一点看看填充效果

        return
    except Exception as e:
        print(f"[严重错误] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


# ========== 后操作（变量集推导、后处理函数-差值、去中、交互） ==========
def derive_var_sets_from_constructs(constructs: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    """Derive variable sets from user-defined `constructs`.

    Returns:
      - static_vars: constructs['demo'] (demographics/background)
      - tp_keys_sorted: sorted timepoint keys ['t0','t1',...]
      - base_time_vars: constructs appearing in >=2 timepoints (for deltas)
      - all_psycho_vars: union of constructs listed in any timepoint (excluding 'demo')
    """
    static_vars = list(constructs.get('demo', []))

    tp_keys = [k for k in constructs.keys() if re.match(r'^t\d+$', k)]
    tp_keys_sorted = sorted(tp_keys, key=lambda x: int(x[1:]))

    per_tp = {tp: [x for x in constructs.get(tp, []) if x != 'demo'] for tp in tp_keys_sorted}

    freq = defaultdict(int)
    for tp, names in per_tp.items():
        for n in names:
            freq[n] += 1
    base_time_vars = sorted([n for n, c in freq.items() if c >= 2])

    all_set = set()
    for tp in tp_keys_sorted:
        for n in per_tp.get(tp, []):
            if n == 'demo':
                continue
            all_set.add(n)
    all_psycho_vars = sorted(all_set)

    return static_vars, tp_keys_sorted, base_time_vars, all_psycho_vars


def apply_post_ops(df_long: pd.DataFrame, constructs: dict, condition: str) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """Recompute post operations on a long table.

    Post operations include:
      - deltas for repeated-measures constructs (base_time_vars)
      - grand-mean centering (_c) and interactions (_inter) for cross-sectional constructs

    This function does NOT modify `constructs` and derives all required variable sets from it.

    Returns:
      (df_final, all_psycho_vars, delta_cols, centered_cols, inter_cols)
    """
    df = df_long.copy()

    # Ensure required id/time columns
    if 'subject_id' not in df.columns:
        raise KeyError("df_long 缺失必需列 'subject_id'")
    if 'time' not in df.columns:
        raise KeyError("df_long 缺失必需列 'time'")

    # Normalize time to int when possible
    df['time'] = pd.to_numeric(df['time'], errors='coerce')

    static_vars, tp_keys_sorted, base_time_vars, all_psycho_vars = derive_var_sets_from_constructs(constructs)

    # Condition coding column name (expects it already exists in psycho.csv / df_long)
    cond01 = f"{condition}01"
    if cond01 not in df.columns:
        # Try fallback: if raw condition exists, create 0/1; otherwise create NA
        if condition in df.columns:
            s = pd.to_numeric(df[condition], errors='coerce')
            uniq = sorted(pd.unique(s.dropna()).tolist())
            if set(uniq).issubset({0, 1}) and len(uniq) <= 2:
                df[cond01] = s.astype('Int64')
            elif len(uniq) >= 2:
                mapping = {uniq[0]: 0, uniq[1]: 1}
                df[cond01] = s.map(mapping).astype('Int64')
            else:
                df[cond01] = pd.Series(pd.NA, index=df.index, dtype='Int64')
        else:
            df[cond01] = pd.Series(pd.NA, index=df.index, dtype='Int64')
            print(f"[警告] df_long 中未找到 '{cond01}'（也未找到 '{condition}'），交互项将为缺失。", file=sys.stderr)

    # ---------- A) Centering & interaction for cross-sectional (single-timepoint) constructs ----------
    exclude_static = {
        'sex', 'age', 'art_exp', 'color_exp', 'art_join', 'ipad_exp', 'health',
        cond01,
    }

    centered_cols: list[str] = []
    inter_cols: list[str] = []

    # cross-sectional psycho vars inferred from constructs (single-timepoint vars)
    cross_sectional_vars = [v for v in all_psycho_vars if v not in base_time_vars and v not in exclude_static]

    cond01_float = pd.to_numeric(df.get(cond01), errors='coerce')

    for v in cross_sectional_vars:
        if v not in df.columns:
            continue

        s_v = pd.to_numeric(df[v], errors='coerce')

        # Subject-level value: first NON-NA value for this variable within subject
        subj_val = s_v.groupby(df['subject_id']).apply(
            lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan
        )
        grand_mean = subj_val.mean(skipna=True)

        c_name = f"{v}_c"
        df[c_name] = s_v - grand_mean
        centered_cols.append(c_name)

        inter_name = f"{v}_inter"
        df[inter_name] = df[c_name] * cond01_float
        inter_cols.append(inter_name)

    # ---------- B) Deltas for repeated-measures constructs ----------
    delta_cols: list[str] = []

    # Ensure numeric for base_time_vars
    for var in base_time_vars:
        if var in df.columns:
            df[var] = df[var].replace(r'^\s*$', pd.NA, regex=True)
            df[var] = pd.to_numeric(df[var], errors='coerce')

    # pivot to wide for delta computation
    wide = df.pivot_table(
        index='subject_id',
        columns='time',
        values=[v for v in base_time_vars if v in df.columns],
        aggfunc='first'
    )

    # Use actual time values present
    time_vals = sorted(df['time'].dropna().astype(int).unique().tolist())

    # Compute and broadcast back to long table by subject_id mapping
    for var in base_time_vars:
        for i, t_from in enumerate(time_vals):
            for t_to in time_vals[i+1:]:
                col_name = f"{var}{t_to}{t_from}"
                delta_cols.append(col_name)

                s_to = wide.get((var, t_to))
                if s_to is None:
                    s_to = pd.Series(np.nan, index=wide.index, dtype='float')

                s_from = wide.get((var, t_from))
                if s_from is None:
                    s_from = pd.Series(np.nan, index=wide.index, dtype='float')

                delta_series = (s_to - s_from)

                # broadcast
                df[col_name] = df['subject_id'].map(delta_series.to_dict())

    # ---------- C) Final column order: required first, then keep any other columns ----------
    required_cols = ['subject_id', 'time'] + all_psycho_vars + delta_cols + static_vars + [cond01] + centered_cols + inter_cols
    other_cols = [c for c in df.columns if c not in required_cols]

    df_final = df.reindex(columns=required_cols + other_cols)
    df_final = df_final.sort_values(by=['subject_id', 'time'])

    return df_final, all_psycho_vars, delta_cols, centered_cols, inter_cols

def main():
    print("--- 心理数据 (SPSS -> 长表) 自动化处理脚本 (v5) ---")
    
    # --- 步骤 1: 从 settings.py 加载配置 ---
    try:
        DS = DATASETS[ACTIVE_DATA]
        paths = DS.get("paths", {})
        CONDI_FILE = (DATA_DIR / paths["groups"] / "groups.csv").resolve()
        OUT_ROOT = (DATA_DIR / paths.get("psycho")).resolve()
        DS = DATASETS[ACTIVE_DATA]
    except KeyError as e:
        print(f"[错误] settings.py 配置有误，缺少键: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[错误] settings.py 配置解析失败: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 步骤 2: 运行模式选择 ---
    # only_cal_total_indices=True：不重新读取原始 t0/t1/t2/t3.csv；仅基于已生成 psycho.csv 重新计算总指标并覆盖。
    # only_cal_total_indices=False：完整流程，读取原始 t0/t1/t2/t3.csv 评分并生成 psycho.csv。

    OUT_FILE = (OUT_ROOT / "psycho.csv").resolve()

    if only_cal_total_indices:
        print("仅更新总指标模式：将直接读取并更新已生成的 psycho.csv，不会要求选择原始问卷文件夹。")
        try:
            update_totals_in_existing_psycho(OUT_FILE)
        except Exception as e:
            print(f"[严重错误] 更新 psycho.csv 总指标失败: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return

    # 完整流程
    print("请选择包含 t0.csv, t1.csv, t2.csv, t3.csv (如有) 的数据文件夹...")
    input_dir = select_folder()

    if input_dir:
        process_data(input_dir, OUT_ROOT, CONDI_FILE)

if __name__ == "__main__":
    main()