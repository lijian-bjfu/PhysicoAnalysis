import pandas as pd
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

def parse_psycho_indices_old(psycho_indices: list):
    """
    按规则，解析 'sttings.psycho_indices' 列表。
    
    它会“发现”时间点，验证它们，并返回处理所需的所有映射。
    
    返回:
        - time_points (list): 排序后的时间点前缀, e.g., ['t0', 't1', 't2']
        - rename_maps (dict): 嵌套字典, e.g., {'t0': {'t0_STAI_Mean': 'STAI_Mean'}, ...}
        - static_vars (list): 非时间性指标, e.g., ['Flow']
        - base_time_vars (list): 基础指标名, e.g., ['STAI_Mean']
    """
    
    time_vars_by_base = defaultdict(list)
    static_vars = []
    all_time_prefixes_found = set() # 存储所有发现的 't<num>'

    # 1. 解析PSYCHO_INDICES，按基础指标名分组
    # 正则表达式匹配 't' + '数字' + '_' + '任意字符'
    regex = re.compile(r"^(t\d+)_(.+)") 

    for var in psycho_indices:
        match = regex.match(var)
        if match:
            prefix = match.group(1)      # e.g., 't1'
            base_name = match.group(2)   # e.g., 'STAI_Mean'
            
            # 存储 (前缀, 原始全名)
            time_vars_by_base[base_name].append((prefix, var)) 
            all_time_prefixes_found.add(prefix)
        else:
            static_vars.append(var) # e.g., 'Flow'

    if not time_vars_by_base:
        # 验证：如果只有静态变量，是否合理
        if not static_vars:
            print("[错误] 'psycho_indices' 列表为空或无法解析。", file=sys.stderr)
            return None, None, None, None
        
        # 只有静态变量，我们假设它们都在 't2.csv' (或某个默认文件)
        # 静态变量是非时间性的，所以它们应附加到最后一个时间点
        # 我们暂时返回一个默认的、最后的时间点 't2'
        print("[警告] 未找到时间性指标。所有指标将作为非时间性处理，并假设从 't2.csv' 加载。", file=sys.stderr)
        time_points = ['t2'] # 默认
        rename_maps = defaultdict(dict, {'t2': {var: var for var in static_vars}})
        return time_points, rename_maps, static_vars, []

    # 2. 验证所有时间性指标是否具有相同的时间点
    base_names = list(time_vars_by_base.keys())
    first_base_name = base_names[0]
    # 获取第一个指标的所有前缀
    expected_prefixes = {p[0] for p in time_vars_by_base[first_base_name]}
    
    for base_name in base_names[1:]:
        current_prefixes = {p[0] for p in time_vars_by_base[base_name]}
        if current_prefixes != expected_prefixes:
            print(f"[错误] 指标的时间点不一致。", file=sys.stderr)
            print(f"    '{first_base_name}' 包含时间点: {expected_prefixes}", file=sys.stderr)
            print(f"    '{base_name}' 却包含: {current_prefixes}", file=sys.stderr)
            print("    所有时间性指标必须在相同的时间点测量。", file=sys.stderr)
            return None, None, None, None
    
    # 3. 应用排序和验证规则
    # expected_prefixes 现在是 {'t0', 't1', 't2'}
    
    # "如果三个数字完全相同则报错"
    if len(expected_prefixes) < 2:
        # (我们假设至少需要两个时间点才算“时间性”)
        # 严格来说，如果 `expected_prefixes` 是 `{'t1'}`，这本身不报错，
        # 但这可能不是一个有效的时间序列。
        # 你的“三个数字相同”规则更可能是指这个。
        print(f"[警告] 仅发现一个时间点: {expected_prefixes}。请检查 'psycho_indices'。", file=sys.stderr)
        
    # 按数字大小排序 (e.g., 't10' 会在 't9' 之后)
    time_points = sorted(list(expected_prefixes), key=lambda p: int(p[1:])) 

    # "如果三个数字为0，1，2，就按照这个顺序排序"
    if time_points == ['t0', 't1', 't2']:
        print("    已识别并排序时间点: ['t0', 't1', 't2'] (符合预期)")
    # "如果三个数字不同，但不是0 1 2，则按照大小排序，但是打印警告"
    else:
        print(f"[警告] 识别的时间点为: {time_points}。", file=sys.stderr)
        print("         期望的时间点为 ['t0', 't1', 't2']。将按数字顺序处理。", file=sys.stderr)

    # 4. 构建返回结果
    rename_maps = defaultdict(dict)
    base_time_vars_set = set()
    
    for base_name, vars_list in time_vars_by_base.items():
        base_time_vars_set.add(base_name)
        for (prefix, full_var_name) in vars_list:
            rename_maps[prefix][full_var_name] = base_name
            
    # "非时间性指标" 附加到最后一个时间点
    if time_points:
        last_time_point = time_points[-1] # e.g., 't2'
        for var in static_vars:
            # 它们不需要重命名
            rename_maps[last_time_point][var] = var
    
    base_time_vars = sorted(list(base_time_vars_set))
    return time_points, rename_maps, static_vars, base_time_vars

def process_data_old(input_dir: Path, psycho_indices: list, out_root_path: Path):
    """
    核心处理逻辑：读取CSV，根据解析结果重塑为长表，并保存。
    """
    try:
        # --- 1. 解析 Settings ---
        print(f"--- 1. 正在解析 'psycho_indices' 列表... ---")
        
        time_points, rename_maps, static_vars, base_time_vars = parse_psycho_indices(psycho_indices)
        
        if time_points is None:
            print("[致命错误] 'psycho_indices' 解析失败，程序终止。", file=sys.stderr)
            return

        OUT_FILE = out_root_path / "psycho.csv"
        out_root_path.mkdir(parents=True, exist_ok=True)
        
        print(f"    将处理的时间点: {time_points}")
        print(f"    将处理的非时间性指标: {static_vars}")
        print(f"    输出文件将保存至: {OUT_FILE}")

        # --- 2. 循环加载所有时间点的CSV文件 ---
        print("\n--- 2. 正在加载CSV文件... ---")
        
        all_dfs = []
        
        # [修改] 使用 enumerate(..., start=1) 来获取 1, 2, 3... 这样的整数索引
        for i, time_point in enumerate(time_points, start=1): # e.g., i=1, time_point='t0'
            file_name = f"{time_point}.csv" # 't0.csv'
            file_path = input_dir / file_name
            
            print(f"    正在读取: {file_path}")

            if not file_path.exists():
                print(f"[警告] 文件缺失: {file_name}，跳过。", file=sys.stderr)
                continue
            
            # 加载CSV
            df = pd.read_csv(file_path)
            # 检查是否存在 'subject_id' 列
            if 'subject_id' not in df.columns:
                raise KeyError(f"文件 '{file_name}' 中缺失 'subject_id' 列，无法对齐数据。")
            
            # 格式化并设置为索引 (以便 pd.concat 自动对齐)
            df['subject_id'] = df['subject_id'].astype(str).str.strip() # 转字符串并去空格
            df = df.set_index('subject_id')
            
            # 添加 'time' 列 (使用整数 1, 2, 3...)
            df['time'] = i # e.g., 1, 2, 3
            
            # 重命名 (动态地!)
            current_rename_map = rename_maps.get(time_point, {})
            df = df.rename(columns=current_rename_map)
            
            all_dfs.append(df)

        print("    所有CSV文件加载成功。")

        # --- 3. 合并为长表 ---
        # join='outer' 确保T2(或最后时间点)的独有列被保留
        df_long = pd.concat(all_dfs, join='outer', ignore_index=False)
        
        # 将 subject_id 从索引变回普通列
        df_long = df_long.reset_index()
        
        print("    已合并为长表。")

        # --- 4. 清理并排序列 ---
        final_cols = ['subject_id', 'time'] + base_time_vars + static_vars
        df_final = df_long.reindex(columns=final_cols)
        
        # 按被试和时间排序
        df_final = df_final.sort_values(by=['subject_id', 'time'])

        # --- 5. 保存到文件 ---
        df_final.to_csv(OUT_FILE, index=False)
        
        print("\n--- 3. 处理完成 ---")
        print(f"    最终长表已成功保存到: {OUT_FILE}")
        print("\n    最终数据预览 (前6行):")
        print(df_final.head(6))
        
    except FileNotFoundError as e:
        print(f"[错误] 文件未找到: {e}。请确保所有在 'psycho_indices' 中定义的时间点 (e.g., {time_points}) 都有对应的CSV文件。", file=sys.stderr)
    except KeyError as e:
        print(f"[错误] 键错误: {e}。很可能 'psycho_indices' 中定义的列名与CSV文件中的列名不匹配。", file=sys.stderr)
    except Exception as e:
        print(f"[严重错误] 处理数据时发生意外错误: {e}", file=sys.stderr)

def parse_psycho_indices(psycho_indices: list):
    """
    解析 'psycho_indices'。
    逻辑更新：
    1. 识别时间点前缀 (t0, t1...)。
    2. 收集所有静态变量 (static_vars)，不绑定特定时间点。
    """
    
    time_vars_by_base = defaultdict(list)
    static_vars = []
    
    # 1. 解析变量名
    regex = re.compile(r"^(t\d+)_(.+)") 
    for var in psycho_indices:
        match = regex.match(var)
        if match:
            prefix = match.group(1)      # e.g., 't1'
            base_name = match.group(2)   # e.g., 'STAI_Mean'
            time_vars_by_base[base_name].append((prefix, var)) 
        else:
            static_vars.append(var) # e.g., 'sex', 'Flow'

    # 2. 收集所有时间点
    base_names = list(time_vars_by_base.keys())
    if base_names:
        expected_prefixes = {p[0] for p in time_vars_by_base[base_names[0]]}
    else:
        expected_prefixes = set()

    # [泛化策略] 如果有静态变量，需要扫描可能存在的 t0-t3 文件。
    # 为了保险，至少确保 t0, t1, t2, t3 (如果有动态变量暗示了最大值) 都被检查。
    # 这里简化处理：如果发现了 t1, t2，通常也应该检查 t0 (基本信息) 和 t3 (结束问卷)。
    # 简单的做法：把 t0 强制加进去（通常基本信息都在这），如果还有其他时间点，后续逻辑会处理。
    if static_vars and 't0' not in expected_prefixes:
        expected_prefixes.add('t0')
        # 如果 Flow 在 t3，且 t3 没有动态变量，这里可能需要手动把 t3 加进去。
        # 但通常 t3 会有状态变量。如果 t3 纯粹只有 Flow，请确保 t3 被加入集合。
        # 假设：用户的数据通常 t1-t3 都有动态指标。如果没有，下面的逻辑也能容错。

    # 3. 排序
    time_points = sorted(list(expected_prefixes), key=lambda p: int(p[1:])) 

    # 4. 构建动态变量映射 (静态变量不再放入 rename_maps)
    rename_maps = defaultdict(dict)
    base_time_vars_set = set()
    
    for base_name, vars_list in time_vars_by_base.items():
        base_time_vars_set.add(base_name)
        for (prefix, full_var_name) in vars_list:
            rename_maps[prefix][full_var_name] = base_name
            
    base_time_vars = sorted(list(base_time_vars_set))
    return time_points, rename_maps, static_vars, base_time_vars

def process_data(input_dir: Path, psycho_indices: list, out_root_path: Path, condi_file: Path | None = None):
    """
    核心处理逻辑：读取CSV，重塑为长表。
    [修改] 泛化处理静态变量：只要CSV里有 settings 定义的静态变量，就保留。
    """
    try:
        print(f"--- 1. 正在解析 'psycho_indices' 列表... ---")
        time_points, rename_maps, static_vars, base_time_vars = parse_psycho_indices(psycho_indices)

        # --- [新增] 清理静态变量：合并/去重 `_Tn` 拷贝，只保留一个并去掉后缀 ---
        # 例如 fss_T3, fss_T2 同时存在时，只保留 n 最大的那个，并重命名为 fss
        # 说明：这一步仅作用于静态变量（非 t0_xxx 这种重复测量规则的变量）。
        _static_best = {}  # base -> (n, original_name)
        _static_plain = set()  # 记录无 _Tn 后缀的静态变量
        _re_t_suffix = re.compile(r"^(?P<base>.+)_T(?P<n>\d+)$")

        for v in list(static_vars):
            m = _re_t_suffix.match(v)
            if m:
                base = m.group('base')
                n = int(m.group('n'))
                prev = _static_best.get(base)
                if prev is None or n > prev[0]:
                    _static_best[base] = (n, v)
            else:
                _static_plain.add(v)

        # 生成：要保留的静态变量 base 名称列表（去掉 _Tn）
        cleaned_static_vars = sorted(list(_static_plain.union(_static_best.keys())))

        # 生成：静态变量重命名映射（仅对被保留的那一列执行 rename）
        static_rename_map = {}
        for base, (_n, orig) in _static_best.items():
            static_rename_map[orig] = base

        # 更新 static_vars 为清理后的 base 名称
        static_vars = cleaned_static_vars

        if static_rename_map:
            kept = [f"{orig}→{base}" for orig, base in static_rename_map.items()]
            print(f"    静态变量已去重并去后缀(_Tn)，保留映射: {kept}")

        if not time_points:
            print("[致命错误] 未找到任何时间点，程序终止。", file=sys.stderr)
            return

        OUT_FILE = out_root_path / "psycho.csv"
        out_root_path.mkdir(parents=True, exist_ok=True)

        print(f"    将扫描的时间点文件: {time_points}")
        print(f"    需要寻找的静态指标: {static_vars}")

        print("\n--- 2. 正在加载CSV文件... ---")
        all_dfs = []

        for time_point in time_points:
            file_name = f"{time_point}.csv"
            file_path = input_dir / file_name

            if not file_path.exists():
                print(f"    [跳过] 文件不存在: {file_name}")
                continue

            print(f"    正在读取: {file_path}")

            # 读取原始数据
            df = pd.read_csv(file_path)

            if 'subject_id' not in df.columns:
                raise KeyError(f"文件 '{file_name}' 中缺失 'subject_id' 列。")

            # 设索引
            df['subject_id'] = df['subject_id'].astype(str).str.strip()
            df = df.set_index('subject_id')

            # 提取时间 time
            try:
                time_num = int(re.search(r'\d+', time_point).group())
            except:
                time_num = 0
            df['time'] = time_num

            # 1. 动态变量重命名
            # 获取当前时间点应该有的动态变量映射 (e.g., {'t1_STAI': 'STAI'})
            current_rename_map = rename_maps.get(time_point, {})
            df = df.rename(columns=current_rename_map)

            # 1.1 静态变量重命名：将 *_Tn 重命名为 base 名（仅对被选中的那一列）
            if static_rename_map:
                df = df.rename(columns=static_rename_map)

            # 2. 筛选列：只保留 (time) + (重命名后的动态变量) + (当前文件里存在的静态变量)

            # A. 想要保留的动态列 (base_time_vars 中存在于当前 df 的)
            cols_to_keep = ['time']
            for col in df.columns:
                if col in current_rename_map.values():
                    cols_to_keep.append(col)

            # B. 想要保留的静态列 (static_vars 中存在于当前 df 的)
            found_static = []
            for static_var in static_vars:
                if static_var in df.columns:
                    cols_to_keep.append(static_var)
                    found_static.append(static_var)

            if found_static:
                print(f"      -> 发现静态指标: {found_static}")

            # 只保留有用列，避免无关数据混入
            df = df[cols_to_keep]

            all_dfs.append(df)

        print("    CSV文件加载完毕。")

        # --- 3. 合并 ---
        # outer join 会自动处理缺失：
        # t0 行会有 sex, age, 但 Flow 为空
        # t3 行会有 Flow, 但 sex, age 为空
        df_long = pd.concat(all_dfs, join='outer', ignore_index=False)
        df_long = df_long.reset_index()
        
        print("    已合并为长表。正在填充静态变量...")

        # --- [新增] 静态变量填充 ---
        # 现在的长表里，t0行有sex没flow，t3行有flow没sex。
        # 需要让同一个 subject_id 的所有行，都拥有这些静态值。
        # 使用 groupby + first/ffill/bfill 来把静态值广播到该人的所有行。
        
        # 对每一个静态变量进行组内填充
        for static_var in static_vars:
            if static_var in df_long.columns:
                # transform('first') 会找到该组第一个非空值，并赋给全组
                # 注意：前提是同一个人的静态变量在不同时间点是一样的（或者只出现一次）
                df_long[static_var] = df_long.groupby('subject_id')[static_var].transform('first')

        # --- [新增] 从分组文件 (groups.csv) 合并条件变量，并编码为 0/1 ---
        cond01 = f"{condition}01"  # e.g., task01

        # groups.csv 预期格式：subject_id,task
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

        # 3) 为所有界面/体验类截面变量生成去中心化版本 (_c) 与交互项 (_inter)
        #    这里默认将 static_vars 中的“界面数据”视为交互候选；排除明显的背景/人口学变量。
        exclude_static = {
            'sex', 'age', 'art_exp', 'color_exp', 'art_join', 'ipad_exp', 'health',
            cond01,
        }

        centered_cols = []
        inter_cols = []

        # 用被试层面均值做 grand-mean centering（避免长表多行重复导致均值被重复计数）
        if static_vars:
            for v in static_vars:
                if v in exclude_static:
                    continue
                if v not in df_long.columns:
                    continue

                # 只对可转换为数值的变量做中心化与交互
                s_v = pd.to_numeric(df_long[v], errors='coerce')
                # 被试层面取 first，再算 grand mean
                subj_mean = s_v.groupby(df_long['subject_id']).first().mean(skipna=True)

                c_name = f"{v}_c"
                df_long[c_name] = s_v - subj_mean
                centered_cols.append(c_name)

                inter_name = f"{v}_inter"
                df_long[inter_name] = df_long[c_name] * df_long[cond01].astype(float)
                inter_cols.append(inter_name)

        # --- 重复测量变量差值 (delta) ---
        # 规则：对每个重复测量变量 var，生成 var{to}{from} = var@time=to - var@time=from
        # 例如 stai21 = stai(t2) - stai(t1), stai32 = stai(t3) - stai(t2), stai31 = stai(t3) - stai(t1)
        delta_cols = []
        if base_time_vars:
            # [修复] 差值计算要求数值型。CSV 中常见的缺失/异常会导致列被读成 object/str，
            # 例如空字符串、"NA"、"nan" 等，进而在做减法时报错。
            # 这里统一将所有重复测量变量强制转为数值；无法解析的值会被置为 NaN。
            for var in base_time_vars:
                if var in df_long.columns:
                    # 先把纯空白转为缺失，再做数值转换
                    df_long[var] = df_long[var].replace(r'^\s*$', pd.NA, regex=True)
                    df_long[var] = pd.to_numeric(df_long[var], errors='coerce')
            # pivot 成宽表，便于按 (var, time) 取值
            wide = df_long.pivot_table(
                index='subject_id',
                columns='time',
                values=base_time_vars,
                aggfunc='first'
            )
            # 使用实际出现过的 time 值来生成差值对（保证与数据一致）
            time_vals = sorted(
                df_long['time'].dropna().astype(int).unique().tolist()
            )

            delta_data = {}
            for var in base_time_vars:
                for i, t_from in enumerate(time_vals):
                    for t_to in time_vals[i+1:]:
                        col_name = f"{var}{t_to}{t_from}"
                        delta_cols.append(col_name)

                        s_to = wide.get((var, t_to))
                        if s_to is None:
                            s_to = pd.Series(float('nan'), index=wide.index)

                        s_from = wide.get((var, t_from))
                        if s_from is None:
                            s_from = pd.Series(float('nan'), index=wide.index)

                        delta_data[col_name] = s_to - s_from

            # 合并回长表：差值变量按 subject_id 广播到所有 time 行
            if delta_data:
                df_deltas = pd.DataFrame(delta_data, index=wide.index).reset_index()
                df_long = df_long.merge(df_deltas, on='subject_id', how='left')

        # --- 4. 清理 ---
        final_cols = ['subject_id', 'time'] + base_time_vars + delta_cols + static_vars + [f"{condition}01"] + centered_cols + inter_cols
        df_final = df_long.reindex(columns=final_cols)
        df_final = df_final.sort_values(by=['subject_id', 'time'])

        # --- 5. 保存 ---
        df_final.to_csv(OUT_FILE, index=False)
        
        print("\n--- 3. 处理完成 ---")
        print(f"    最终长表保存至: {OUT_FILE}")
        print(df_final.head(10)) # 打印多一点看看填充效果
        
    except Exception as e:
        print(f"[严重错误] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

def main():
    print("--- 心理数据 (SPSS -> 长表) 自动化处理脚本 (v5) ---")
    
    # --- 步骤 1: 从 settings.py 加载配置 ---
    try:
        DS = DATASETS[ACTIVE_DATA]
        paths = DS.get("paths", {})
        PSYCHO_INDICES = DS.get("psycho_indices", [])
        CONDI_FILE = (DATA_DIR / paths["groups"] / "groups.csv").resolve()
        OUT_ROOT = (DATA_DIR / paths.get("psycho")).resolve()
        DS = DATASETS[ACTIVE_DATA]
    
    except KeyError as e:
        print(f"[错误] settings.py 配置有误，缺少键: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[错误] settings.py 配置解析失败: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 步骤 2: 让用户选择CSV数据文件夹 ---
    print(f"请选择包含 {', '.join([f'{t}.csv' for t in parse_psycho_indices(PSYCHO_INDICES)[0]])} 的数据文件夹...")
    input_dir = select_folder()
    
    if input_dir:
        # --- 步骤 3: 执行核心处理逻辑 ---
        process_data(input_dir, PSYCHO_INDICES, OUT_ROOT, CONDI_FILE)

if __name__ == "__main__":
    main()