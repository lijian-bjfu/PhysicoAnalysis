# scripts/loaders/fantasia_loader.py
import pandas as pd
import numpy as np
import wfdb
from pathlib import Path

# --- helpers ---
def _resolve_root(data_dir: Path, root_str: str) -> Path:
    p = Path(root_str)
    return p if p.is_absolute() else (data_dir / p)

def _discover_records(root: Path, options: dict) -> list[str]:
    """优先从已缓存的 *_ecg.parquet 推断被试；否则从 .hea 推断；否则用 options['records']。"""
    cached = sorted({p.stem.split("_")[0] for p in root.glob("*_ecg.parquet")})
    if cached:
        return cached
    heads = sorted({p.stem for p in root.glob("*.hea")})
    if heads:
        return heads
    return options.get("records", []) or []

def _save_continuous(sid: str, sig: str, values: np.ndarray, fs: float, root: Path, cache_fmt: str) -> Path:
    n = len(values)
    t = np.arange(n, dtype=float) / float(fs)
    df = pd.DataFrame({"time_s": t, "value": values.astype(float), "fs_hz": float(fs)})
    out = root / f"{sid}_{sig}.{ 'parquet' if cache_fmt=='parquet' else 'csv' }"
    if cache_fmt == "parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    return out

def _summary_of_cached(path: Path) -> tuple[float, float, int]:
    df = pd.read_parquet(path) if path.suffix==".parquet" else pd.read_csv(path)
    dur = float(df["time_s"].max()) if len(df) else 0.0
    fs  = float(df["fs_hz"].iloc[0]) if "fs_hz" in df.columns and len(df) else float("nan")
    return fs, dur, len(df)

# --- main entry ---
def load(dataset_cfg: dict, data_dir: Path) -> pd.DataFrame:
    """
    保存到 DATASETS['fantasia']['root'] 下：
      <sid>_ecg.parquet  (time_s,value,fs_hz)
      <sid>_resp.parquet (time_s,value,fs_hz)
    并返回一个轻量 summary DataFrame（不会拼巨大长表）。
    """
    root    = _resolve_root(data_dir, dataset_cfg["root"])
    opts    = dataset_cfg.get("options", {}) or {}
    signals = opts.get("signals", ["ecg","resp"])
    allow_net = bool(opts.get("allow_network", False))
    prefer_local = bool(opts.get("prefer_local_wfdb", True))
    cache_fmt = str(opts.get("cache_format", "parquet")).lower()

    root.mkdir(parents=True, exist_ok=True)

    # 发现记录
    records = _discover_records(root, opts)
    if not records:
        print("[fantasia] 没找到任何记录。请在 settings.DATASETS['fantasia']['root'] 放置 .hea/.dat 或已有 *_ecg.parquet。")
        return pd.DataFrame(columns=["subject_id","signal","source","fs_hz","dur_s","n_rows","path"])

    print(f"[fantasia] found {len(records)} records under {root}")

    first_subject_schema_printed = False
    rows = []

    for i, sid in enumerate(records, 1):
        print(f"[fantasia] [{i}/{len(records)}] subject '{sid}'")

        # 1) 缓存优先：若已存在 *_ecg.parquet/_resp.parquet 就直接读取概览
        cached_any = False
        for sig in signals:
            cached = root / f"{sid}_{sig}.parquet"
            if cached.exists():
                cached_any = True
                fs, dur, n = _summary_of_cached(cached)
                print(f"          cached → {cached}  (fs={fs:.0f} Hz, dur≈{dur:.0f}s, n={n})")
                rows.append({"subject_id": sid, "signal": sig, "source": "cache",
                             "fs_hz": fs, "dur_s": dur, "n_rows": n, "path": str(cached)})

        # 如果所需信号全都已有缓存，且已经打印了，跳过解析
        if all((root / f"{sid}_{sig}.parquet").exists() for sig in signals):
            # 只对第一个被试打印一次“列命名提示”
            if not first_subject_schema_printed:
                print("          [schema] columns for cached continuous signals: ['time_s','value','fs_hz']")
                first_subject_schema_printed = True
            continue

        # 2) 本地 WFDB：有 .hea/.dat/.ecg 就解析
        parsed = False
        if prefer_local:
            hea = root / f"{sid}.hea"
            dat = root / f"{sid}.dat"
            if hea.exists() and dat.exists():
                try:
                    rec = wfdb.rdrecord(str(root / sid))
                    fs = float(rec.fs)
                    # 挑通道：名字里含 ECG/RESP 的
                    name_map = {nm.upper(): idx for idx, nm in enumerate(rec.sig_name)}
                    for want in signals:
                        if want == "ecg":
                            ch = None
                            for k in name_map:
                                if "ECG" in k:
                                    ch = name_map[k]; break
                            if ch is not None:
                                arr = rec.p_signal[:, ch]
                                out = _save_continuous(sid, "ecg", arr, fs, root, cache_fmt)
                                fs2, dur2, n2 = _summary_of_cached(out)
                                print(f"          parsed(local) → {out}  (fs={fs2:.0f} Hz, dur≈{dur2:.0f}s, n={n2})")
                                rows.append({"subject_id": sid, "signal": "ecg", "source": "wfdb_local",
                                             "fs_hz": fs2, "dur_s": dur2, "n_rows": n2, "path": str(out)})
                                parsed = True
                        elif want == "resp":
                            ch = None
                            for k in name_map:
                                if "RESP" in k:
                                    ch = name_map[k]; break
                            if ch is not None:
                                arr = rec.p_signal[:, ch]
                                out = _save_continuous(sid, "resp", arr, fs, root, cache_fmt)
                                fs2, dur2, n2 = _summary_of_cached(out)
                                print(f"          parsed(local) → {out}  (fs={fs2:.0f} Hz, dur≈{dur2:.0f}s, n={n2})")
                                rows.append({"subject_id": sid, "signal": "resp", "source": "wfdb_local",
                                             "fs_hz": fs2, "dur_s": dur2, "n_rows": n2, "path": str(out)})
                                parsed = True
                except Exception as e:
                    print(f"          [warn] wfdb local parse failed for '{sid}': {e}")

        # 3) 联网（可选）
        if (not parsed) and allow_net:
            try:
                rec = wfdb.rdrecord(sid, pn_dir="fantasia")
                fs = float(rec.fs)
                name_map = {nm.upper(): idx for idx, nm in enumerate(rec.sig_name)}
                for want in signals:
                    if want == "ecg":
                        ch = None
                        for k in name_map:
                            if "ECG" in k:
                                ch = name_map[k]; break
                        if ch is not None:
                            arr = rec.p_signal[:, ch]
                            out = _save_continuous(sid, "ecg", arr, fs, root, cache_fmt)
                            fs2, dur2, n2 = _summary_of_cached(out)
                            print(f"          parsed(network) → {out}  (fs={fs2:.0f} Hz, dur≈{dur2:.0f}s, n={n2})")
                            rows.append({"subject_id": sid, "signal": "ecg", "source": "network",
                                         "fs_hz": fs2, "dur_s": dur2, "n_rows": n2, "path": str(out)})
                            parsed = True
                    elif want == "resp":
                        ch = None
                        for k in name_map:
                            if "RESP" in k:
                                ch = name_map[k]; break
                        if ch is not None:
                            arr = rec.p_signal[:, ch]
                            out = _save_continuous(sid, "resp", arr, fs, root, cache_fmt)
                            fs2, dur2, n2 = _summary_of_cached(out)
                            print(f"          parsed(network) → {out}  (fs={fs2:.0f} Hz, dur≈{dur2:.0f}s, n={n2})")
                            rows.append({"subject_id": sid, "signal": "resp", "source": "network",
                                         "fs_hz": fs2, "dur_s": dur2, "n_rows": n2, "path": str(out)})
                            parsed = True
            except Exception as e:
                print(f"          [warn] wfdb network parse failed for '{sid}': {e}")

        # 只在第一个被试打印一次“列命名说明”
        if not first_subject_schema_printed and (cached_any or parsed):
            print("          [schema] columns for continuous signals: ['time_s','value','fs_hz']")
            first_subject_schema_printed = True

    # 收尾：打印轻量 summary（每个被试×信号一行）
    df_sum = pd.DataFrame(rows)
    if not df_sum.empty:
        # 汇总每个 subject 的可用信号
        for sid, sub in df_sum.groupby("subject_id"):
            parts = [f"{r['signal']}: fs={r['fs_hz']:.0f}Hz, dur≈{r['dur_s']:.0f}s, n={int(r['n_rows'])}"
                     for _, r in sub.iterrows()]
            print(f"[ok] {sid} | " + " | ".join(parts))
    return df_sum