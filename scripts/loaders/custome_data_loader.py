# 读取自己的CSV：最小列要求 time_s,value,fs_hz,subject_id,signal
from pathlib import Path
import pandas as pd
import glob

def load(dataset_cfg, data_dir: Path) -> pd.DataFrame:
    pattern = dataset_cfg["csv_glob"]  # 例如 "data/raw/custom/*.csv"
    files = glob.glob(pattern)
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    required = {"time_s","value","fs_hz","subject_id","signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV缺少必要列: {missing}")
    # 可选：加 age_group 列
    if "age_group" not in df.columns:
        df["age_group"] = "unknown"
    return df