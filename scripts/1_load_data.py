# scripts/1_load_data.py
import importlib, sys, pandas as pd
from pathlib import Path

# bootstrap
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent

from settings import DATASETS, ACTIVE_DATA, PROJECT_ROOT, PROCESSED_DIR

def _to_abs(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

def main():
    # 取这一次要用的数据集配置
    dataset_cfg = DATASETS[ACTIVE_DATA].copy()
    dataset_cfg["root"]   = _to_abs(dataset_cfg["root"])
    if dataset_cfg.get("events"):
        dataset_cfg["events"] = _to_abs(dataset_cfg["events"])
    else:
        dataset_cfg["events"] = None

    loader_mod = dataset_cfg["loader"]
    mod = importlib.import_module(loader_mod)

    print(f"[load] dataset='{ACTIVE_DATA}'  loader='{loader_mod}'")
    print(f"[load] root → {dataset_cfg['root']}")
    if dataset_cfg["events"]:
        print(f"[load] events → {dataset_cfg['events']}")

    # 让 loader 自己去归位并返回摘要
    summary: pd.DataFrame = mod.load(dataset_cfg)

    out = PROCESSED_DIR / f"1_summary_{ACTIVE_DATA}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"[save] 摘要 → {out} (rows={len(summary)})")

    # 给点肉眼可见的东西
    with pd.option_context("display.max_rows", 20, "display.max_columns", 10):
        print(summary.head(20))

if __name__=="__main__":
    main()