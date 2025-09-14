# scripts/1_load_data.py
import importlib
import pandas as pd

# --- project-root bootstrap ---
import sys
from pathlib import Path
_p = Path(__file__).resolve()
for _ in range(6):  # 最多向上爬6层
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent))
        break
    _p = _p.parent
# --- end bootstrap ---
from settings import DATASETS, DATA_DIR, PROCESSED_DIR, VERBOSE

# 加载一阈值（1000万行）
MAX_COMBINED_ROWS = 10_000_000

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        mod_name = f"scripts.loaders.{ds['source']}_loader"
        if VERBOSE:
            print(f"[loader] use → {mod_name}  dataset='{ds['name']}'")
        module = importlib.import_module(mod_name)

        df = module.load(ds, DATA_DIR)

        summary = df.groupby(["subject_id","signal"]).agg(
            dur_s=("time_s","max"),
            fs_hz=("fs_hz","max"),
            n_samples=("time_s","size"),
        ).reset_index()

        # 浏览 summary
        out_summary = PROCESSED_DIR / f"1dis_{ds['name']}_summary.csv"
        summary.to_csv(out_summary, index=False)
        if VERBOSE:
            print(f"[save] summary → {out_summary}")
        
        # 再按阈值决定要不要写大表（由于全库数据过大，默认只读 MAX_COMBINED_ROWS 之内数量的数据）
        if len(df) <= MAX_COMBINED_ROWS:
            out_parquet = PROCESSED_DIR / f"1dis_{ds['name']}_raw.parquet"
            df.to_parquet(out_parquet, index=False)
            if VERBOSE:
                print(f"[save] long table → {out_parquet} (rows={len(df)})")
        else:
            if VERBOSE:
                print(f"[skip] long table skipped (rows={len(df)} > {MAX_COMBINED_ROWS}). "
                    "Use per-record parquet in data/raw/physionet/fantasia/")

        # 打印数据特征
        for sid, sub in summary.groupby("subject_id"):
            lines = []
            for _, r in sub.iterrows():
                lines.append(f"{r['signal']}: fs={r['fs_hz']:.0f}Hz, dur≈{r['dur_s']:.0f}s, n={int(r['n_samples'])}")
            print(f"[ok] {sid} | " + " | ".join(lines))

if __name__ == "__main__":
    main()