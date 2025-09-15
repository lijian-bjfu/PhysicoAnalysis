# scripts/loaders/custome_data_loader.py
import sys, shutil, re
import pandas as pd
from pathlib import Path

# bootstrap
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent

from settings import SIGNAL_ALIASES
from scripts.standard.naming import infer_sid

def _ask_dir() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        d = filedialog.askdirectory(title="选择原始导出目录")
        if not d: raise RuntimeError("未选择目录")
        return Path(d)
    except Exception:
        p = input("源目录路径：").strip()
        return Path(p)

def _detect_signal_by_name(name: str) -> str|None:
    low = name.lower()
    for sig, keys in SIGNAL_ALIASES.items():
        if any(k in low for k in keys):
            return sig
    return None

def _hr_acc_vendor_suffix(name: str) -> str:
    low = name.lower()
    if "h10" in low:    return "h"   # H10
    if "verity" in low: return "v"   # Verity
    return ""                        # 未知就不加

# --- helpers ---
def _resolve_root(data_dir: Path, root_str: str) -> Path:
    p = Path(root_str)
    return p if p.is_absolute() else (data_dir / p)

def load(dataset_cfg: dict, data_dir: Path) -> pd.DataFrame:
    """把用户选定目录的文件归位到 settings['root']/settings['events']。不读内容。"""
    root   = _resolve_root(data_dir, dataset_cfg.get("root",   "data/raw/local"))
    events = _resolve_root(data_dir, dataset_cfg.get("events", "data/raw/local/events")) \
             if dataset_cfg.get("events") else None
             
    opts         = dataset_cfg.get("options", {})
    ask_dir      = bool(opts.get("ask_dir", True))
    copy_mode    = str(opts.get("copy_mode", "copy2")).lower()

    root.mkdir(parents=True, exist_ok=True)
    if events: events.mkdir(parents=True, exist_ok=True)

    src_dir = _ask_dir() if ask_dir else Path(input("源目录路径：").strip())
    print(f"[local] 源目录：{src_dir}")

    rows = []
    files = sorted([p for p in src_dir.rglob("*") if p.is_file()])
    if not files:
        print("[local] 目录为空？你选的文件夹里没有东西。")
        return pd.DataFrame(columns=["subject_id","signal","dest","status","note","bytes"])

    for p in files:
        name = p.name
        sig  = _detect_signal_by_name(name)
        if sig is None:
            rows.append({"subject_id":"","signal":"","dest":"","status":"skip","note":"unrecognized", "bytes":p.stat().st_size})
            continue

        # sid 从文件名里尽力猜；猜不到就 P001S001T001R001（但我会提醒你）
        sid = infer_sid(name) or "P001S001T001R001"

        # hr/acc 的厂商后缀（hhr/vhr/hacc/vacc）
        vendor = _hr_acc_vendor_suffix(name)
        sig_out = sig
        if sig == "hr" and vendor:
            sig_out = f"{vendor}hr"
        if sig == "acc" and vendor:
            sig_out = f"{vendor}acc"

        # 事件类单独目录，其他都进 root
        dest_dir = events if (sig == "events" and events) else root
        ext = p.suffix.lower() if p.suffix else ""
        if ext == "": ext = ".csv"   # 没后缀就强行补一个，省得乱套

        dest = dest_dir / f"{sid}_{sig_out}{ext}"

        # 搬运（不读内容）
        try:
            if copy_mode == "move":
                shutil.move(str(p), str(dest))
            elif copy_mode == "copy":
                shutil.copy(str(p), str(dest))
            else:
                shutil.copy2(str(p), str(dest))
            rows.append({"subject_id":sid, "signal":sig_out, "dest":str(dest), "status":"ok", "note":"", "bytes":dest.stat().st_size})
            print(f"[ok] {name} → {dest.relative_to(dest_dir.parent)}")
        except Exception as e:
            rows.append({"subject_id":sid, "signal":sig_out, "dest":str(dest), "status":"fail", "note":str(e), "bytes":p.stat().st_size})
            print(f"[fail] {name}: {e}")

    # 返回摘要：不读内容，不“分析”
    df = pd.DataFrame(rows)
    # 小结统计
    if not df.empty:
        total = int(df["bytes"].sum())
        okn   = int((df["status"]=="ok").sum())
        print(f"[local] 完成：ok={okn}/{len(df)}  total_size={total/1e6:.1f} MB")
    return df