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

def parse_psycho_indices(psycho_indices: list):
    """
    按你的规则，解析 'psycho_indices' 列表。
    
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
            prefix = match.group(1)      # e.g., 't0'
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
        # 根据你的逻辑，这是非时间性的，所以它们应附加到最后一个时间点
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


def process_data(input_dir: Path, psycho_indices: list, out_root_path: Path):
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
            
            # 加载CSV
            df = pd.read_csv(file_path, index_col=0).rename_axis("subject_id")
            df.index = df.index.astype(str) # 确保索引为字符串
            
            # [修改] 添加 'time' 列 (使用整数 1, 2, 3...)
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

def main():
    print("--- 心理数据 (SPSS -> 长表) 自动化处理脚本 (v5) ---")
    
    # --- 步骤 1: 从 settings.py 加载配置 ---
    try:
        DS = DATASETS[ACTIVE_DATA]
        paths = DS.get("paths", {})
        PSYCHO_INDICES = DS.get("psycho_indices", [])
        OUT_ROOT = (DATA_DIR / paths.get("psycho")).resolve()
    
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
        process_data(input_dir, PSYCHO_INDICES, OUT_ROOT)

if __name__ == "__main__":
    main()