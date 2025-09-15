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

def _resolve_under_data_dir(p: str|Path) -> Path:
    """仅用于打印展示：把相对路径显示成挂在 DATA_DIR 下的绝对路径。"""
    pp = Path(p)
    return pp if pp.is_absolute() else (DATA_DIR / pp)

def main():
    # 取这一次要用的数据集配置（不做就地修改，交给 loader 自己解析）
    dataset_cfg = DATASETS[ACTIVE_DATA]

    loader_mod = dataset_cfg["loader"]
    mod = importlib.import_module(loader_mod)

    print(f"[load] dataset='{ACTIVE_DATA}'  loader='{loader_mod}'")
    if "root" in dataset_cfg:
        print(f"[load] root   → {_resolve_under_data_dir(dataset_cfg['root'])}")
    if dataset_cfg.get("events"):
        print(f"[load] events → {_resolve_under_data_dir(dataset_cfg['events'])}")

    # 统一签名：把 DATA_DIR 传给 loader，由它来解析相对/绝对路径
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