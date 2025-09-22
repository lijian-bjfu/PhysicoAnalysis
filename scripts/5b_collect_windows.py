from __future__ import annotations
import sys, shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent))
        break
    _p = _p.parent
# --------------------------------------------------------
from settings import DATASETS, ACTIVE_DATA, DATA_DIR

DS = DATASETS[ACTIVE_DATA]
paths = DS["paths"]
SRC_ROOT = (DATA_DIR / paths["windowing"]).resolve()                 # e.g. data/processed/windowing/local
OUT_ROOT = (SRC_ROOT / "collected").resolve()                        # e.g. .../windowing/local/collected
OUT_ROOT.mkdir(parents=True, exist_ok=True)
APPLY_TO: List[str] = DS.get("windowing", {}).get("apply_to", ["rr"])

PRE_SID = DATASETS[ACTIVE_DATA]["preview_sids"]

# ----------------- helpers -----------------
def _fmt_hms(sec: float) -> str:
    """Format seconds to H:MM:SS.mmm (human-friendly)."""
    if pd.isna(sec):
        return "NaN"
    sec = float(sec)
    sign = "-" if sec < 0 else ""
    sec = abs(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{sign}{h:d}:{m:02d}:{s:06.3f}"

def _parse_level(p: Path) -> Optional[int]:
    """Return level number from a folder named like 'level3'."""
    name = p.name.lower()
    if name.startswith("level"):
        try:
            return int(name.replace("level", ""))
        except ValueError:
            return None
    return None

def _scan_levels(src_root: Path) -> List[Tuple[int, Path]]:
    """Find all levelN folders under SRC_ROOT, sorted by level number ascending."""
    levels: List[Tuple[int, Path]] = []
    for child in src_root.iterdir():
        if child.is_dir():
            lv = _parse_level(child)
            if lv is not None:
                levels.append((lv, child))
    levels.sort(key=lambda x: x[0])
    return levels

def _load_index(level_path: Path, level_no: int) -> pd.DataFrame:
    """Load level/index.csv and add level column (int)."""
    idx_file = level_path / "index.csv"
    if not idx_file.exists():
        raise FileNotFoundError(f"缺少 index.csv: {idx_file}")
    df = pd.read_csv(idx_file)
    # ensure required cols exist
    required = ["subject_id", "w_id", "t_start_s", "t_end_s"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{idx_file} 缺少必要列: {c}")
    # add inferred columns if missing
    if "level" not in df.columns:
        df["level"] = level_no
    else:
        # normalize to int
        df["level"] = df["level"].astype(int)
    for c in ["parent_level", "parent_w_id"]:
        if c not in df.columns:
            df[c] = 0
    if "meaning" not in df.columns:
        df["meaning"] = ""
    if "lineage_path" not in df.columns:
        # build a simple lineage seed if absent
        df["lineage_path"] = [f"level{level_no}/w{int(w):02d}" for w in df["w_id"]]
    # types
    df["subject_id"] = df["subject_id"].astype(str)
    df["w_id"] = df["w_id"].astype(int)
    df["parent_level"] = df["parent_level"].astype(int)
    df["parent_w_id"] = df["parent_w_id"].astype(int)
    df["t_start_s"] = pd.to_numeric(df["t_start_s"], errors="coerce")
    df["t_end_s"] = pd.to_numeric(df["t_end_s"], errors="coerce")
    # sanity: start <= end
    bad = df["t_end_s"] < df["t_start_s"]
    if bad.any():
        raise ValueError(f"{idx_file} 存在结束时间早于起始时间的窗口: 行 {df.index[bad].tolist()}")
    return df

def _collect_all_index(src_root: Path) -> pd.DataFrame:
    """Concat all level*/index.csv into one long table."""
    levels = _scan_levels(src_root)
    if not levels:
        raise SystemExit(f"[error] 未发现任何 level* 目录: {src_root}")
    frames = []
    print(f"[collect] 发现 {len(levels)} 个层级: {[lv for lv, _ in levels]}")
    for lv, p in levels:
        df = _load_index(p, lv)
        df["src_level_dir"] = str(p)
        frames.append(df)
    all_idx = pd.concat(frames, ignore_index=True)
    # uniqueness check
    dup = all_idx.duplicated(subset=["subject_id", "level", "w_id"], keep=False)
    if dup.any():
        rows = all_idx.loc[dup, ["subject_id", "level", "w_id"]].drop_duplicates()
        raise ValueError(f"[error] 同一层级重复 w_id: \n{rows}")
    return all_idx

def _visible_leaf_windows(idx_all: pd.DataFrame, subjects: List[str]) -> pd.DataFrame:
    """Compute final visible (leaf) windows per subject."""
    idx = idx_all.copy()
    if subjects:
        idx = idx[idx["subject_id"].isin(subjects)].copy()
    # parents set = nodes referenced by children
    parent_keys = set(zip(idx["subject_id"], idx["parent_level"], idx["parent_w_id"]))
    node_keys = list(zip(idx["subject_id"], idx["level"], idx["w_id"]))
    is_parent = [k in parent_keys for k in node_keys]
    idx["is_parent"] = is_parent
    visible = idx[~idx["is_parent"]].copy()
    # final order per subject by time
    visible["duration_s"] = visible["t_end_s"] - visible["t_start_s"]
    visible.sort_values(["subject_id", "t_start_s", "t_end_s"], inplace=True)
    visible["final_order"] = visible.groupby("subject_id").cumcount() + 1
    return visible

def _find_src_window_file(level_dir: Path, sid: str, signal: str, w_id: int) -> Optional[Path]:
    """Find a source window file by trying csv then parquet."""
    fname_csv = f"{sid}_{signal}_w{int(w_id):02d}.csv"
    fname_parq = f"{sid}_{signal}_w{int(w_id):02d}.parquet"
    p_csv = level_dir / fname_csv
    p_parq = level_dir / fname_parq
    if p_csv.exists():
        return p_csv
    if p_parq.exists():
        return p_parq
    return None

def _copy_visible_files(visible: pd.DataFrame, out_root: Path, src_root: Path, apply_to: List[str]) -> pd.DataFrame:
    """Copy all visible window files into collected/<signal>/..., build rename_map table."""
    recs = []
    # prepare per-signal folder
    for sig in apply_to:
        (out_root / sig).mkdir(parents=True, exist_ok=True)
    # We need a mapping from (subject_id, level) to actual level dir path
    level_dirs: Dict[Tuple[str, int], Path] = {}
    # We stored src level dir string in idx_all; keep a lookup
    # Build from visible rows
    for _, r in visible.iterrows():
        level_dirs[(r["subject_id"], int(r["level"]))] = Path(str(r["src_level_dir"]))

    subs = sorted(visible["subject_id"].unique().tolist())
    for sid in tqdm(subs, desc="拷贝可见窗口", unit="subject"):
        sub_df = visible[visible["subject_id"] == sid].copy()
        for _, r in sub_df.iterrows():
            level_dir = level_dirs[(sid, int(r["level"]))]
            w_id = int(r["w_id"])
            order = int(r["final_order"])
            for sig in apply_to:
                src = _find_src_window_file(level_dir, sid, sig, w_id)
                if src is None:
                    print(f"[skip] {sid} {sig} w{w_id:02d}: 源文件缺失（level{int(r['level'])}）")
                    continue
                dst = (out_root / sig / f"{sid}_{sig}_w{order:02d}{src.suffix}").resolve()
                shutil.copyfile(src, dst)
                recs.append({
                    "subject_id": sid,
                    "signal": sig,
                    "final_order": order,
                    "src_level": int(r["level"]),
                    "src_w_id": w_id,
                    "src_path": str(src),
                    "dst_path": str(dst)
                })
    return pd.DataFrame.from_records(recs)

# ----------------- window-axis helper -----------------
def _attach_window_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Add window-axis columns w_s and w_e per subject.
    Rules:
      - Anchor per subject at earliest t_start_s among FINAL visible windows.
      - w_s = round(t_start_s - t0, 2)
      - w_e = round(w_s + round(t_end_s - t_start_s, 2), 2)
      - Unit is seconds; keep 2 decimals. Snap tiny -0.00 to +0.00.
    """
    res = df.copy()
    # ensure numeric
    res["t_start_s"] = pd.to_numeric(res["t_start_s"], errors="coerce")
    res["t_end_s"] = pd.to_numeric(res["t_end_s"], errors="coerce")
    # per-subject anchor
    t0 = res.groupby("subject_id")["t_start_s"].transform("min")
    # compute w_s, snap near-zero to 0.00 to avoid -0.00 artifacts
    w_s = np.round(res["t_start_s"] - t0, 2)
    w_s = np.where(np.abs(w_s) < 5e-3, 0.0, w_s)
    # use rounded duration to keep (w_e - w_s) numerically consistent with CSV rounded values
    w_len = np.round(res["t_end_s"] - res["t_start_s"], 2)
    w_e = np.round(w_s + w_len, 2)
    # assign with stable float dtype
    res["w_s"] = w_s.astype(float)
    res["w_e"] = w_e.astype(float)
    return res

# ----------------- main -----------------
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[collect] 输入目录（windowing 根）：{SRC_ROOT}")
    print(f"[collect] 输出目录（collected）：{OUT_ROOT}")
    print(f"[collect] 信号集合 apply_to={APPLY_TO}")

    # 1) merge all index
    idx_all = _collect_all_index(SRC_ROOT)

    # --- DEBUG: 确认源索引是否已注入 src_level_dir ---
    print(f"[debug] idx_all columns = {list(idx_all.columns)} | nrows={len(idx_all)}")
    if "src_level_dir" not in idx_all.columns:
        raise RuntimeError(
            "idx_all 缺少 src_level_dir。请确认 _collect_all_index 是否按最新版注入，"
            "以及 level*/index.csv 是否可读。若刚升级脚本，建议先删掉旧的 collected/ 再重跑。"
        )

    # 2) subject 选择
    all_subjects = sorted(idx_all["subject_id"].unique().tolist())

    # 选择所有被试，若想预览一下，用 PRE_SID
    subjects = all_subjects
    print(f"[select] 被试总数 {len(all_subjects)}，将处理 {len(subjects)} → {subjects}")

    # 3) 计算可见叶子窗 + 最终顺序
    visible = _visible_leaf_windows(idx_all, subjects)

    print(f"[debug] visible(before merge) cols = {list(visible.columns)} | nrows={len(visible)}")

    # 关联源 level 目录，稳健处理：优先从 idx_all 合并；若失败或全空，则按 level→目录 映射兜底
    if "src_level_dir" not in visible.columns:
        if "src_level_dir" in idx_all.columns:
            visible = visible.merge(
                idx_all[["subject_id", "level", "w_id", "src_level_dir"]],
                on=["subject_id", "level", "w_id"],
                how="left",
                validate="1:1"
            )
        else:
            # 极端情况：上游没注入该列，则直接按 level 推断目录
            level_map = {lv: str(p) for lv, p in _scan_levels(SRC_ROOT)}
            visible["src_level_dir"] = visible["level"].map(level_map)

    # 若合并后仍然缺失或全是 NaN，则继续按 level 推断目录
    if "src_level_dir" not in visible.columns or visible["src_level_dir"].isna().all():
        level_map = {lv: str(p) for lv, p in _scan_levels(SRC_ROOT)}
        visible["src_level_dir"] = visible["level"].map(level_map)

    print(f"[debug] visible(after attach src_level_dir) has column? { 'src_level_dir' in visible.columns }")

    # 4) 输出 collected_index.csv
    collected_index = visible[[
        "subject_id", "final_order", "level", "w_id",
        "parent_level", "parent_w_id",
        "t_start_s", "t_end_s", "duration_s",
        "meaning", "lineage_path", "src_level_dir"
    ]].copy()

    # attach window-axis (w_s, w_e) per subject, 2-decimal seconds anchored at the first final window
    collected_index = _attach_window_axis(collected_index)

    collected_index.sort_values(["subject_id", "final_order"], inplace=True)
    out_index = OUT_ROOT / "collected_index.csv"
    collected_index.to_csv(out_index, index=False)
    print(f"[save] collected_index.csv → {out_index} (rows={len(collected_index)})")

    # 5) 复制所有可见窗口文件到 collected/<signal> 并生成 rename_map.csv
    rename_map = _copy_visible_files(collected_index, OUT_ROOT, SRC_ROOT, APPLY_TO)
    out_map = OUT_ROOT / "rename_map.csv"
    rename_map.to_csv(out_map, index=False)
    print(f"[save] rename_map.csv → {out_map} (rows={len(rename_map)})")

    # 6) 写人类可读的 log.txt（仅一次，总览，不逐被试罗列）
    out_log = OUT_ROOT / "log.txt"
    with out_log.open("w", encoding="utf-8") as f:
        f.write("【收集汇总日志（cohort 级）】\n")
        f.write(f"输入窗口根目录：{SRC_ROOT}\n")
        f.write(f"输出目录：{OUT_ROOT}\n")
        f.write(f"参与信号集合：{APPLY_TO}\n")
        levels = [lv for lv, _ in _scan_levels(SRC_ROOT)]
        f.write(f"发现层级：{levels}\n")
        f.write(f"被试数量：{len(subjects)}（详见 collected_index.csv 的 subject_id 列）\n")
        f.write(f"最终可见窗口总数：{len(collected_index)}\n")
        # 每层窗口数量统计（非叶与叶都报，便于审计）
        try:
            level_counts = idx_all.groupby(["level"]).size().reset_index(name="n_windows")
            f.write("各层级窗口数量（含父窗与叶窗）：\n")
            for _, r in level_counts.iterrows():
                f.write(f"  level{int(r['level'])}: {int(r['n_windows'])} 个\n")
        except Exception:
            pass
        f.write("\n命名与追溯规则：\n")
        f.write("  1) 最终窗按每位被试的时间顺序重排，重命名为 w01, w02, ...，文件位于 collected/<signal>/ 下。\n")
        f.write("  2) rename_map.csv 记录了每个文件从源路径到目标路径的映射，便于追溯。\n")
        f.write("  3) collected_index.csv 保存了每个最终窗的元数据（level, w_id, parent_level, parent_w_id, lineage_path 等）。\n")
        f.write("  4) lineage/parent 语义：若某窗在更高层被细分，则其为父窗，不会进入最终集合；叶窗为最终参与分析的单位。\n")
        f.write("  5) 新增时间轴列：w_s、w_e。以每位被试的首个最终窗口起点为原点(0)，单位为秒，保留2位小数；如果真实数据早于此原点，则对应对齐时可出现负值。\n")
        f.write("\n说明：\n")
        f.write("  本日志仅记录流程与汇总信息，不逐一罗列各被试窗口明细，以避免冗余。\n")
        f.write("  若需查看某被试的最终窗口，请查阅 collected_index.csv 并配合 rename_map.csv。\n")
    print(f"[save] log.txt → {out_log}")

    print("[done] 收集完成。你现在可以用 collected_index.csv 与 rename_map.csv 去对接特征计算。")

if __name__ == "__main__":
    main()
