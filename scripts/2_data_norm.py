# scripts/2_data_norm.py
"""
批处理数据规范化（超薄壳）：
- 读取 settings.DATASETS[ACTIVE_DATA] 中的 root 目录
- 递归扫描文件（csv/parquet 为主；其它交给各 loader 预处理）
- 调用标准化小工具：
    - scripts.standard.naming.infer_sid / detect_signal
    - scripts.standard.relabel.map_to_standard
    - scripts.standard.schema.to_continuous / to_rr
- 统一落盘到 PROCESSED_DIR / norm / <ACTIVE_DATA> 下
- 同时写 10 行 preview CSV，便于肉眼复核
"""

import sys
from pathlib import Path
import pandas as pd

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATASETS, ACTIVE_DATA, DATA_DIR, PROCESSED_DIR, SIGNAL_ALIASES

# 只当“薄壳”，所有脑力外包给小工具
from scripts.standard.naming import infer_sid, detect_signal
from scripts.standard.relabel import map_to_standard
from scripts.standard.schema import to_continuous, to_rr, to_hr

# 支持的原始后缀（统一由 loader 负责把奇葩格式转成这些）
RAW_EXTS = {".csv", ".parquet"}

# 打印详细信息的开关
VERBOSE = True

def _resolve_root(root_str: str) -> Path:
    p = Path(root_str)
    return p if p.is_absolute() else (DATA_DIR / Path(root_str))

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    # 容错 CSV
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, sep=r"\s+", engine="python")

def _save_standard(std: pd.DataFrame,
                   out_dir: Path,
                   sid: str,
                   sig: str,
                   make_preview: bool = True) -> tuple[Path, str]:
    """
    方案A：根据原始 sig 决定“形状保障 + 文件名 + 扩展名”，一次性完成。
    返回：(最终文件路径, 最终信号名)
      - 连续信号(ecg/resp/ppg/acc) → to_continuous → <sid>_<sig>.parquet
      - 逐搏(rr/ppi)                → to_rr         → <sid>_rr.csv
      - 心率(hr)                     → to_hr         → <sid>_hr.csv
    """
    sig_low = sig.lower().strip()
    if sig_low in ("rr", "ppi"):
        final_sig = "rr"
        out_df = to_rr(std)
        out_path = out_dir / f"{sid}_{final_sig}.csv"
        out_df.to_csv(out_path, index=False)
        prev_cols = ["t_s", "rr_ms"]
    elif sig_low == "hr":
        final_sig = "hr"
        out_df = to_hr(std)
        out_path = out_dir / f"{sid}_{final_sig}.csv"
        out_df.to_csv(out_path, index=False)
        prev_cols = ["time_s", "hr_bpm"]
    elif sig_low in ("ecg", "resp", "ppg", "acc"):
        final_sig = sig_low
        out_df = to_continuous(std)
        out_path = out_dir / f"{sid}_{final_sig}.parquet"
        out_df.to_parquet(out_path, index=False)
        # 连续信号常用列
        prev_cols = [c for c in ["time_s", "value", "fs_hz"] if c in out_df.columns]
    elif sig_low == "events":
        # 你要求：events 完全跳过
        raise ValueError("events 被显式跳过，不应进入 _save_standard")
    else:
        raise ValueError(f"未支持的信号类型：{sig}")

    # 可选：产 10 行预览 csv，方便肉眼复核
    if make_preview:
        prev_dir = out_dir / "preview"
        prev_dir.mkdir(parents=True, exist_ok=True)
        prev = out_df[prev_cols].head(10) if prev_cols else out_df.head(10)
        prev.to_csv(prev_dir / f"{sid}_{final_sig}_preview.csv", index=False)

    return out_path, final_sig

def main():
    ds = DATASETS[ACTIVE_DATA]
    root = _resolve_root(ds["root"])
    options = ds.get("options", {}) or {}

    out_dir = PROCESSED_DIR / "norm" / ACTIVE_DATA
    out_dir.mkdir(parents=True, exist_ok=True)

    if VERBOSE:
        print(f"[norm] dataset='{ACTIVE_DATA}'  loader='{ds.get('loader')}'")
        print(f"[norm] root → {root}")
        print(f"[norm] out  → {out_dir}")

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in RAW_EXTS]
    if not files:
        print(f"[warn] 在 {root} 下没发现可处理文件（{', '.join(RAW_EXTS)}）。先用对应 loader 把原始数据归位。")
        return


    rows = []
    seen = set()  # 去重：避免同一 <sid,sig> 被 csv/parquet 各处理一次
    total = len(files)

    for i, path in enumerate(sorted(files), 1):
        # 只靠文件名识别信号；events 在此就跳过
        sig = detect_signal(path.name, SIGNAL_ALIASES)
        if not sig:
            if VERBOSE: print(f"[skip] 无法识别信号类型：{path.name}")
            continue
        if sig == "events":
            if VERBOSE: print(f"[skip] 跳过 events：{path.name}")
            continue

        # 从文件名推断被试 ID；支持 BIDS & 自定义 pattern
        sid = infer_sid(path.name, pattern=options.get("sid_pattern"))

        # 同一 <sid,sig> 已经有一个版本落盘了，则跳过重复（例如同名 csv + parquet）
        key = (sid, "rr" if sig in ("rr", "ppi") else ("hr" if sig == "hr" else sig))
        if key in seen:
            if VERBOSE: print(f"[dup] 已处理过 {key}，跳过 {path.name}")
            continue

        try:
            raw = _read_any(path)
            std = map_to_standard(sig, raw, options=options)  # 仅列名与单位标准化
            out_p, final_sig = _save_standard(std, out_dir, sid, sig, make_preview=True)
            seen.add(key)

            if VERBOSE:
                cols = ",".join(std.columns.tolist())
                print(f"[{i}/{total}] {path.name} → {out_p.name}  (rows={len(std)})  [{cols}]")

            rows.append({
                "source_file": str(path.relative_to(root)),
                "subject_id": sid,
                "signal": final_sig,
                "rows": len(std)
            })

        except Exception as e:
            print(f"[fail] {path.name}: {e}")
            continue

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(out_dir / "norm_summary.csv", index=False)
        print(f"[save] 概览 → {out_dir / 'norm_summary.csv'}")
    else:
        print("[warn] 没有任何文件被规范化。检查 detect_signal/infer_sid 或原始目录。")

if __name__ == "__main__":
    main()