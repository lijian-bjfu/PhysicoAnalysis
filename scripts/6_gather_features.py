import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATASETS, ACTIVE_DATA, DATA_DIR, SIGNAL_ALIASES, SCHEMA, PARAMS

# Dataset config
DS = DATASETS[ACTIVE_DATA]
paths = DS["paths"]
SRC_DIR = (DATA_DIR / paths["windowing"] / "collected").resolve()
OUT_ROOT = (DATA_DIR / paths["features"]).resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Signals and feature plan
APPLY_TO: List[str] = DS.get("windowing", {}).get("apply_to", ["rr"])  # e.g., ["rr", "resp"]
SIGNAL_FEATURES = DS.get("signal_features", [])    # e.g., ["mean_hr_bpm","hf_ms2","rsa_ms"]

# Feature modules (aliases to avoid name collision)
try:
    from scripts.features import hrv_time as f_time
    from scripts.features import hrv_freq as f_freq
    from scripts.features import hrv_rsa  as f_rsa
except ImportError:
    from features import hrv_time as f_time
    from features import hrv_freq as f_freq
    from features import hrv_rsa  as f_rsa

print(f"[INFO] hrv_time module: {getattr(f_time, '__file__', 'unknown')}")
print(f"[INFO] hrv_freq module: {getattr(f_freq, '__file__', 'unknown')}")
print(f"[INFO] hrv_rsa module: {getattr(f_rsa, '__file__', 'unknown')}")

# Regex for <sid>_<sig>_wNN.csv
FILE_RE = re.compile(r"^(?P<sid>[^_]+)_(?P<sig>[^_]+)_w(?P<wid>\d+)\.csv$")


def _read_index(index_dir: Path) -> pd.DataFrame:
    """Read collected index and normalize to columns:
       subject_id, w_id (final order), meaning, w_s, w_e
    """
    idx_path = index_dir / "collected_index.csv"
    if not idx_path.exists():
        # fallback if user used a shorter name by mistake
        alt = index_dir / "collect_index.csv"
        if alt.exists():
            idx_path = alt
        else:
            raise FileNotFoundError(f"collected_index.csv not found at {index_dir}")
    df = pd.read_csv(idx_path)
    # prefer final_order as the window id used in filenames; else fall back to w_id if that is the final order
    key_col = "final_order" if "final_order" in df.columns else "w_id"
    need_core = ["subject_id", key_col, "meaning"]
    missing = [c for c in need_core if c not in df.columns]
    if missing:
        raise ValueError(f"{idx_path.name} 缺少必要列: {missing}")
    out = df[need_core].copy()
    out = out.rename(columns={key_col: "w_id"})
    # optional w_s / w_e
    for c in ("w_s", "w_e"):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = pd.NA
    # normalize types
    out["subject_id"] = out["subject_id"].astype(str)
    try:
        out["w_id"] = out["w_id"].astype(int)
    except Exception:
        out["w_id"] = pd.to_numeric(out["w_id"], errors="coerce").astype("Int64")
    return out


def _standardize_df(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    """按 SCHEMA 统一列名到 ['t_s', value_col]。RR 的值列统一为 'rr_ms'；呼吸为 'resp'。"""
    if signal not in SCHEMA:
        raise KeyError(f"SCHEMA 中不存在信号定义: {signal}")
    t_col = SCHEMA[signal].get("t")
    v_col = SCHEMA[signal].get("v")
    if t_col not in df.columns or v_col not in df.columns:
        raise KeyError(f"文件缺少列: 期望[{t_col},{v_col}]，实际列={list(df.columns)}")
    out = pd.DataFrame({"t_s": df[t_col].astype(float).to_numpy()})
    if signal == "rr":
        out["rr_ms"] = df[v_col].astype(float).to_numpy()
    elif signal == "resp":
        out["resp"] = df[v_col].astype(float).to_numpy()
    else:
        # 其他信号暂不处理
        out[v_col] = df[v_col].to_numpy()
    return out


def _load_segment(sid: str, sig: str, wid: int) -> Optional[pd.DataFrame]:
    sig_dir = SRC_DIR / sig
    if not sig_dir.exists() or not sig_dir.is_dir():
        return None
    fname = f"{sid}_{sig}_w{wid:02d}.csv"
    fpath = (sig_dir / fname)
    if not fpath.exists():
        # 兼容不补零的命名
        fname = f"{sid}_{sig}_w{wid}.csv"
        fpath = (sig_dir / fname)
        if not fpath.exists():
            return None
    try:
        df = pd.read_csv(fpath)
        return _standardize_df(df, sig)
    except Exception as e:
        print(f"[WARN] 读取失败：{fpath.name} -> {e}")
        return None


# --- RR window QC (minimal) -------------------------------------------------
def _qc_rr_window(rr_df: pd.DataFrame, w_s: Optional[float], w_e: Optional[float]) -> Dict[str, Optional[float]]:
    """Compute per-window RR quality metrics with minimal scope.
    Returns two fields:
      - rr_valid_ratio: sum(rr_ms)/window_length_in_seconds, clipped to [0,1.0] if window length is finite
      - rr_max_gap_s:   max(diff(t_s)) within the window (seconds)
    If inputs are insufficient, returns NaN for the metric(s).
    """
    # Validate window length
    try:
        win_len = float(w_e) - float(w_s)
    except Exception:
        win_len = np.nan
    if not np.isfinite(win_len) or win_len <= 0:
        win_len = np.nan

    # Clean RR table
    if rr_df is None or rr_df.empty:
        return {"rr_valid_ratio": np.nan, "rr_max_gap_s": np.nan}
    rr = rr_df.copy()
    if "t_s" not in rr.columns or "rr_ms" not in rr.columns:
        return {"rr_valid_ratio": np.nan, "rr_max_gap_s": np.nan}
    rr["t_s"] = pd.to_numeric(rr["t_s"], errors="coerce")
    rr["rr_ms"] = pd.to_numeric(rr["rr_ms"], errors="coerce")
    rr = rr.dropna(subset=["t_s", "rr_ms"])  # keep only valid rows

    # rr_max_gap_s
    if len(rr) > 1:
        try:
            rr_max_gap_s = float(np.nanmax(np.diff(rr["t_s"].to_numpy(float))))
        except Exception:
            rr_max_gap_s = np.nan
    else:
        rr_max_gap_s = np.nan

    # rr_valid_ratio
    if np.isfinite(win_len) and len(rr) > 0:
        try:
            coverage_s = float(rr["rr_ms"].sum() / 1000.0)
            # Clip coverage to window length if finite
            if np.isfinite(win_len):
                coverage_s = float(np.clip(coverage_s, 0.0, win_len))
            rr_valid_ratio = coverage_s / win_len if np.isfinite(win_len) and win_len > 0 else np.nan
        except Exception:
            rr_valid_ratio = np.nan
    else:
        rr_valid_ratio = np.nan

    # Optional rounding for stable CSV appearance
    if np.isfinite(rr_valid_ratio):
        rr_valid_ratio = round(float(rr_valid_ratio), 3)
    if np.isfinite(rr_max_gap_s):
        rr_max_gap_s = round(float(rr_max_gap_s), 3)

    return {"rr_valid_ratio": rr_valid_ratio, "rr_max_gap_s": rr_max_gap_s}


def main():
    import time
    start_time = time.time()

    idx = _read_index(SRC_DIR)  # subject_id, w_id, meaning, w_s, w_e

    # 先扫描 rr 文件，目录改为 SRC_DIR / 'rr'
    rr_dir = SRC_DIR / "rr"
    if not rr_dir.exists() or not rr_dir.is_dir():
        print(f"[WARN] 未找到 rr 信号子目录: {rr_dir}")
        rr_files = []
    else:
        rr_files = [m for m in rr_dir.iterdir() if m.is_file() and FILE_RE.match(m.name) and FILE_RE.match(m.name).group("sig") == "rr"]
        if not rr_files:
            print(f"[WARN] 未在 {rr_dir} 找到任何 rr 窗口文件。")

    # 计算 subjects 和 windows
    subjects = set()
    windows = set()
    for f in rr_files:
        m = FILE_RE.match(f.name)
        subjects.add(m.group("sid"))
        windows.add(int(m.group("wid")))
    subjects = sorted(subjects)
    windows = sorted(windows)

    # 扫描所有信号文件，求可用信号，改为遍历 SRC_DIR 下的信号子目录
    signals_found = set()
    if SRC_DIR.exists() and SRC_DIR.is_dir():
        for d in SRC_DIR.iterdir():
            if d.is_dir() and d.name in APPLY_TO:
                signals_found.add(d.name)
    signals_to_apply = sorted(signals_found.intersection(set(APPLY_TO)))

    # Handle SIGNAL_FEATURES as list, coerce legacy dict if needed
    if isinstance(SIGNAL_FEATURES, dict):
        # legacy dict: union all values to list
        requested_cols = []
        for v in SIGNAL_FEATURES.values():
            if isinstance(v, list):
                requested_cols.extend(v)
            else:
                requested_cols.append(v)
        requested_cols = sorted(set(requested_cols))
    elif isinstance(SIGNAL_FEATURES, list):
        requested_cols = SIGNAL_FEATURES
    else:
        requested_cols = []

    TIME_COLS = {"mean_hr_bpm","rmssd_ms","sdnn_ms","pnn50_pct","sd1_ms","sd2_ms"}
    FREQ_COLS = {"hf_ms2","hf_log_ms2","lf_ms2","lf_log_ms2","hf_band_used","hf_center_hz"}
    RSA_COLS  = {"rsa_ms","resp_rate_bpm","n_breaths_used","rsa_method"}

    rr_plan = []
    if set(requested_cols).intersection(TIME_COLS):
        rr_plan.append("time")
    if set(requested_cols).intersection(FREQ_COLS):
        rr_plan.append("freq")
    if set(requested_cols).intersection(RSA_COLS):
        rr_plan.append("rsa")

    # 打印运行配置和数据集摘要，增加打印实际存在的信号子目录
    existing_signal_dirs = [str(SRC_DIR / sig) for sig in APPLY_TO if (SRC_DIR / sig).exists() and (SRC_DIR / sig).is_dir()]
    print(f"[INFO] 输入目录: {SRC_DIR}")
    print(f"[INFO] 存在的信号子目录: {existing_signal_dirs}")
    print(f"[INFO] 输出目录: {OUT_ROOT}")
    print(f"[INFO] 受试者数量: {len(subjects)}")
    print(f"[INFO] 窗口数量: {len(windows)}")
    print(f"[INFO] 应用信号: {signals_to_apply}")
    print(f"[INFO] rr 文件数量: {len(rr_files)}")
    print(f"[INFO] 请求的特征列: {requested_cols}")
    print(f"[INFO] 推断的 rr 特征计划: {rr_plan}")
    print(f"[INFO] PARAMS 设置: use_individual_hf={PARAMS.get('use_individual_hf', False)}, log_power={PARAMS.get('log_power', False)}")

    combos = []  # (sid, wid)
    for p in rr_files:
        m = FILE_RE.match(p.name)
        sid = m.group("sid")
        wid = int(m.group("wid"))
        combos.append((sid, wid))
    combos = sorted(set(combos), key=lambda x: (x[0], x[1]))

    rows = []

    # Counters
    n_rows = 0
    n_time = 0
    n_freq = 0
    n_rsa = 0
    n_rsa_skipped_no_resp = 0
    n_missing_rr = 0
    n_read_errors = 0
    n_skipped_no_index = 0

    for sid, wid in tqdm(combos, desc="Computing features", unit="win"):
        rr_df = _load_segment(sid, "rr", wid)
        if rr_df is None or rr_df.empty:
            print(f"[WARN] 缺少 RR 段：{sid}_rr_w{wid}")
            n_missing_rr += 1
            continue

        resp_df = None
        need_resp = ("rsa" in rr_plan) or bool(PARAMS.get("use_individual_hf", False))
        if need_resp:
            resp_df = _load_segment(sid, "resp", wid)
            if resp_df is None or resp_df.empty:
                print(f"[WARN] 需要呼吸段但未找到：{sid}_resp_w{wid}（将跳过 RSA 与个体化HF）")
                resp_df = None
                n_rsa_skipped_no_resp += 1

        # per-window metadata from collected_index (meaning, w_s, w_e)
        meta = idx[(idx["subject_id"] == sid) & (idx["w_id"] == wid)]
        if meta.empty:
            print(f"[WARN] 无切窗索引信息，跳过：{sid}_w{wid:02d}")
            n_skipped_no_index += 1
            continue
        meta_row = meta.iloc[0]

        feat_parts = []
        if "time" in rr_plan:
            try:
                feat_parts.append(f_time.features_segment(rr_df))
                n_time += 1
            except Exception as e:
                print(f"[WARN] 时域特征失败 {sid}/w{wid}: {e}")
                n_read_errors += 1
        if "freq" in rr_plan:
            try:
                feat_parts.append(f_freq.features_segment(rr_df, resp_df=resp_df))
                n_freq += 1
            except Exception as e:
                print(f"[WARN] 频域特征失败 {sid}/w{wid}: {e}")
                n_read_errors += 1
        if "rsa" in rr_plan:
            try:
                if resp_df is None:
                    feat_parts.append(pd.DataFrame([[pd.NA, pd.NA, 0, "unavailable"]],
                                                  columns=["rsa_ms","resp_rate_bpm","n_breaths_used","rsa_method"]))
                else:
                    feat_parts.append(f_rsa.features_segment(rr_df, resp_df=resp_df))
                n_rsa += 1
            except Exception as e:
                print(f"[WARN] RSA 特征失败 {sid}/w{wid}: {e}")
                n_read_errors += 1

        feat_df = pd.concat(feat_parts, axis=1) if feat_parts else pd.DataFrame()

        # per-window RR QC (minimal): rr_valid_ratio, rr_max_gap_s
        qc = _qc_rr_window(rr_df, meta_row.get("w_s", np.nan), meta_row.get("w_e", np.nan))

        row = {
            "subject_id": sid,
            "w_id": wid,
            "w_s": float(meta_row["w_s"]) if pd.notna(meta_row["w_s"]) else pd.NA,
            "w_e": float(meta_row["w_e"]) if pd.notna(meta_row["w_e"]) else pd.NA,
            "meaning": meta_row.get("meaning", "")
        }
        for c in requested_cols:
            if c in feat_df.columns:
                row[c] = feat_df.iloc[0][c]
            else:
                row[c] = pd.NA

        # attach QC metrics
        row.update(qc)

        rows.append(row)
        n_rows += 1

    out_df = pd.DataFrame(rows)
    out_path = OUT_ROOT / "physico_features.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] 写出特征表: {out_path}  共 {len(out_df)} 行 × {out_df.shape[1]} 列")

    elapsed = time.time() - start_time
    print(f"[SUMMARY] 处理完成，耗时 {elapsed:.1f} 秒")
    print(f"  总窗口数: {len(combos)}")
    print(f"  成功计算窗口数: {n_rows}")
    print(f"  缺少 RR 窗口数: {n_missing_rr}")
    print(f"  无切窗索引信息跳过数: {n_skipped_no_index}")
    print(f"  时域特征成功数: {n_time}")
    print(f"  频域特征成功数: {n_freq}")
    print(f"  RSA 特征成功数: {n_rsa}")
    print(f"  RSA 跳过(无呼吸段)数: {n_rsa_skipped_no_resp}")
    print(f"  读取/计算错误数: {n_read_errors}")


if __name__ == "__main__":
    main()