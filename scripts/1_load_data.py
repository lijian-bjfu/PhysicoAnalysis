# scripts/1_load_data.py
# scripts/1_load_data.py
import importlib
import pandas as pd
from pathlib import Path
import sys

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATA_DIR, PROCESSED_DIR, DATASETS, ACTIVE_DATA

def main():
    # 取这一次要用的数据集配置（不做就地修改，交给 loader 自己解析）
    dataset_cfg = DATASETS[ACTIVE_DATA]
    all_paths = dataset_cfg.get("paths", {})

    loader_mod = dataset_cfg["loader"]
    
    print(f"[load] dataset='{ACTIVE_DATA}'  loader='{loader_mod}'")
    if "raw" in all_paths:
        print(f"[load] raw data path   →{(DATA_DIR / all_paths['raw']).resolve()}")
    # if "events" in all_paths:
    #     print(f"[load] events data path  → {(DATA_DIR / all_paths['events']).resolve()}")
    # else:
    #     print(f"[load] events data path →{ACTIVE_DATA} 数据集没有实验事件标记数据")

    # 统一签名：把 DATA_DIR 传给 loader，由它来解析相对/绝对路径
    mod = importlib.import_module(loader_mod)
    summary: pd.DataFrame = mod.load(dataset_cfg, DATA_DIR)

    # 保存摘要
    out = PROCESSED_DIR / f"1_summary_{ACTIVE_DATA}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"[save] 摘要 → {out} (rows={len(summary)})")

    # 终端预览前几行
    with pd.option_context("display.max_rows", 20, "display.max_columns", 10):
        print(summary.head(20))

if __name__ == "__main__":
    main()