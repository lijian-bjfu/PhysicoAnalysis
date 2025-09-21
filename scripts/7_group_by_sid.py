import pandas as pd
from pathlib import Path
import sys
import os
from typing import Dict, Any

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATA_DIR, DATASETS, ACTIVE_DATA


def _get_groups_cfg(ds: Dict[str, Any]) -> Dict[str, Any]:
    """兼容 'groups' 与旧写法 'gourps' 两种键名。"""
    if "groups" in ds:
        return ds["groups"]
    if "gourps" in ds:  # 容错：历史拼写
        return ds["gourps"]
    raise KeyError("DATASETS[ACTIVE_DATA] 未配置 'groups' / 'gourps' 区块")


def _scan_subjects(src_dir: Path) -> list[str]:
    if not src_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {src_dir}")
    files = [f for f in os.listdir(src_dir) if (src_dir / f).is_file()]
    sids = set()
    for fname in files:
        stem = fname.rsplit('.', 1)[0]
        sid = stem.split('_')[0]  # 去掉信号后缀
        if sid:
            sids.add(sid)
    return sorted(sids)


def _parse_by_slices(sid: str, group_map: Dict[str, Dict[str, int]], emit_fields: list[str]) -> Dict[str, str]:
    row: Dict[str, str] = {}
    for field in emit_fields:
        spec = group_map.get(field)
        if spec is None:
            raise ValueError(f"group_map 中缺少字段 {field}")
        # 允许 tuple/list 或 dict 写法
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            start, end = spec
        elif isinstance(spec, dict):
            start, end = int(spec.get("start", 0)), int(spec.get("end", 0))
        else:
            raise ValueError(f"group_map[{field}] 非法：{spec}")
        row[field] = sid[start:end]
    return row


def main() -> None:
    # Dataset config
    DS = DATASETS[ACTIVE_DATA]
    paths = DS["paths"]
    SRC_DIR = (DATA_DIR / paths["norm"]).resolve()
    OUT_ROOT = (DATA_DIR / paths["groups"]).resolve()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    groups_cfg = _get_groups_cfg(DS)
    group_map = groups_cfg.get("group_map")
    if not isinstance(group_map, dict) or not group_map:
        raise ValueError("settings.groups.group_map 未配置或为空")

    use = groups_cfg.get("use")
    if use and isinstance(use, list) and len(use) > 0:
        emit_fields = use
    else:
        emit_fields = list(group_map.keys())

    value_map = groups_cfg.get("value", {})

    # 打印配置摘要
    print("[INFO] 组解析配置：mode=slices (按位置切片)")
    print(f"[INFO] 使用字段 (use): {emit_fields}")
    for k in emit_fields:
        spec = group_map.get(k)
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            start, end = spec
        else:
            start, end = spec.get("start"), spec.get("end")
        print(f"  - {k}: start={start}, end={end}")
    print(f"[INFO] 最终输出的组列: {emit_fields}")

    # 扫描被试
    print(f"[INFO] 输入目录: {SRC_DIR}")
    print(f"[INFO] 输出目录: {OUT_ROOT}")
    sids = _scan_subjects(SRC_DIR)
    if not sids:
        print("[WARN] 未在输入目录中找到任何文件，groups.csv 将为空。")
    print(f"[INFO] 发现被试数量: {len(sids)}")

    # 解析
    parsed_rows = []
    skipped_sids = []
    for sid in sids:
        row_raw = _parse_by_slices(sid, group_map, emit_fields)
        row_mapped = {}
        skip = False
        for field in emit_fields:
            raw_val = row_raw.get(field)
            field_map = value_map.get(field)
            if field_map is None:
                # No mapping for this field, just skip mapping and keep raw string
                # But per instructions, if no mapping or missing raw, skip this sid
                skip = True
                break
            mapped_val = field_map.get(raw_val)
            if mapped_val is None:
                skip = True
                break
            row_mapped[field] = mapped_val
        if skip:
            print(f"[WARN] 被试 {sid} 因映射缺失被跳过。")
            skipped_sids.append(sid)
            continue
        row = {"subject_id": sid}
        row.update(row_mapped)
        parsed_rows.append(row)

    df = pd.DataFrame(parsed_rows)
    # 重排列顺序，不插入 group_id 列
    cols = ["subject_id"] + emit_fields
    df = df[cols]

    # 统计分组数量，按每个字段统计
    for field in emit_fields:
        n_levels = df[field].nunique(dropna=False)
        print(f"[INFO] 列 '{field}' 共解析出 {n_levels} 个水平")

    # 写出
    out_path = OUT_ROOT / "groups.csv"
    df.to_csv(out_path, index=False)

    # 预览
    print("[OK] groups.csv 已生成")
    print(f"[OK] 保存路径: {out_path}")
    print("[PREVIEW]")
    print(df.head())

    # 跳过的被试统计
    if skipped_sids:
        print(f"[WARN] 共跳过 {len(skipped_sids)} 个被试，因映射缺失。")
        for sid in skipped_sids:
            print(f"  - {sid}")


if __name__ == "__main__":
    main()