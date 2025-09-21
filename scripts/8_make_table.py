import pandas as pd
import sys
from pathlib import Path
from typing import Tuple

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATASETS, ACTIVE_DATA, DATA_DIR

DS = DATASETS[ACTIVE_DATA]

def _paths() -> Tuple[Path, Path, Path]:
    DS = DATASETS[ACTIVE_DATA]
    paths = DS["paths"]
    dat_dir = (DATA_DIR / paths["features"]).resolve()
    grp_dir = (DATA_DIR / paths["groups"]).resolve()
    out_root = (DATA_DIR / paths["final"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    return dat_dir, grp_dir, out_root

def main() -> None:
    DAT_DIR, GRP_DIR, OUT_ROOT = _paths()

    feats_path = DAT_DIR / "physico_features.csv"
    groups_path = GRP_DIR / "groups.csv"

    print(f"[INFO] 特征表: {feats_path}")
    print(f"[INFO] 分组表: {groups_path}")
    print(f"[INFO] 输出目录: {OUT_ROOT}")

    if not feats_path.exists():
        raise FileNotFoundError(f"未找到特征表 physico_features.csv: {feats_path}")
    if not groups_path.exists():
        raise FileNotFoundError(f"未找到分组表 groups.csv: {groups_path}")

    # 读取
    feats = pd.read_csv(feats_path)
    groups = pd.read_csv(groups_path)

    # 基本校验
    if "subject_id" not in feats.columns:
        raise KeyError("特征表缺少列 'subject_id'")
    if "subject_id" not in groups.columns:
        raise KeyError("分组表缺少列 'subject_id'")

    use_cols = DS.get('groups', {}).get('use', None)
    candidate_cols = [c for c in groups.columns if c != 'subject_id']
    if use_cols and isinstance(use_cols, list) and len(use_cols) > 0:
        group_cols = [c for c in use_cols if c in candidate_cols]
        missing_cols = [c for c in use_cols if c not in candidate_cols]
        if missing_cols:
            print(f"[WARN] 配置中请求的分组列缺失: {missing_cols}")
    else:
        group_cols = candidate_cols

    if not group_cols:
        raise KeyError("分组表中未找到任何分组列。请检查 groups.csv 和配置。")

    groups = groups[['subject_id'] + group_cols]

    # 去重：同一个 subject 取第一条分组
    if groups.duplicated(subset=["subject_id"]).any():
        dup_n = int(groups.duplicated(subset=["subject_id"]).sum())
        print(f"[WARN] 分组表中发现 {dup_n} 个重复 subject_id，保留首次出现的记录。")
        groups = groups.drop_duplicates(subset=["subject_id"], keep="first")

    # 合并（左连接，保持特征表所有行）
    before_rows = len(feats)
    final = feats.merge(groups, on="subject_id", how="left")

    # 每个分组列的缺失统计
    for col in group_cols:
        missing_count = int(final[col].isna().sum())
        if missing_count > 0:
            miss_ids = final.loc[final[col].isna(), "subject_id"].unique().tolist()
            print(f"[WARN] 分组列 '{col}' 有 {missing_count} 条缺失，涉及 {len(miss_ids)} 位被试：{miss_ids[:10]}{' ...' if len(miss_ids)>10 else ''}")

    # 写出
    out_path = OUT_ROOT / "final.csv"
    final.to_csv(out_path, index=False)

    # 报告
    print(f"[OK] 合并完成：{before_rows} → {len(final)} 行，列数 {final.shape[1]}，合并分组列: {group_cols}")
    print(f"[OK] 已写出: {out_path}")
    print("[PREVIEW] 最前 5 行：")
    print(final.head())

if __name__ == "__main__":
    main()