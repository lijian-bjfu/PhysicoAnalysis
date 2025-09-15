# scripts/2a_select_rr.py
import numpy as np, pandas as pd, sys
import matplotlib.pyplot as plt
from pathlib import Path

# boot
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
from settings import RAW_CACHE_DIR, PROCESSED_DIR, RR_COMPARE, PARAMS

SEL_DIR = PROCESSED_DIR / "rr_select"; SEL_DIR.mkdir(parents=True, exist_ok=True)
DEC_SUGG = SEL_DIR / "decision_suggested.csv"
DEC_FILE = SEL_DIR / "decision.csv"

# --- 用户过滤配置（支持短号） ---------------------------------
# 例子：
# SUBJECTS_FILTER = []                  # 空：处理全部
# SUBJECTS_FILTER = ["P001S001T001R001"]# 指定完整 sid
# SUBJECTS_FILTER = ["001","002"]       # 短号：匹配 P001…、P002…
SUBJECTS_FILTER = ["001"]  # 你要的示例；用完记得清空或改

# 限制最多预览多少个（None 表示不限）
PREVIEW_LIMIT = None
# ---------------------------------------------------------------

def _read_ecg(sid: str):
    p = RAW_CACHE_DIR / f"{sid}_ecg.parquet"
    if not p.exists(): p = RAW_CACHE_DIR / f"{sid}_ecg.csv"
    if not p.exists(): return None
    df = pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
    return df

def _ecg_to_rr(ecg_df: pd.DataFrame):
    import neurokit2 as nk
    df = ecg_df.copy()

    # 1) 先把时间轴和振幅整理干净
    if "time_s" not in df.columns or "value" not in df.columns:
        raise ValueError("ECG 缺少 time_s 或 value 列")

    # 排序 + 去重 + 去 NaN
    df = df.dropna(subset=["time_s", "value"]).sort_values("time_s")
    # 去掉完全相同时间戳的重复点，避免 dt 中位数被拉成 0
    dup_mask = df["time_s"].diff().fillna(0).eq(0)
    if dup_mask.any():
        df = df[~dup_mask]

    t = df["time_s"].to_numpy(float)
    sig = df["value"].to_numpy(float)
    n = len(sig)

    # 2) 采样率稳健估计：优先用列里的 fs_hz；不靠谱就用 dt 中位数反推
    fs_col = df.get("fs_hz", pd.Series([np.nan])).iloc[0]
    fs = float(fs_col) if pd.notna(fs_col) else np.nan
    if not np.isfinite(fs) or fs < 10 or fs > 1000:
        dt = np.median(np.diff(t)) if n > 2 else np.nan
        fs_est = (1.0 / dt) if (dt is not None and np.isfinite(dt) and dt > 0) else np.nan
        fs = float(fs_est) if np.isfinite(fs_est) else 130.0  # 兜底 130 Hz

    # NeuroKit 的平滑核会按 fs 算，fs 太低会出 0 核 -> 报错。强行提到 >=10。
    if fs < 10:
        print(f"[warn] ECG fs_hz={fs:.2f}Hz 太低，改为 130Hz 再做 R 峰检测")
        fs = 130.0

    # 如果信号太短，直接放弃
    if n < int(3 * fs):  # 少于约 3 秒，别逞强
        print(f"[warn] ECG 长度只有 {n} 点(<~3s)，放弃 ECG→RR")
        return None

    # 3) 试跑 ecg_process；若失败改用更稳的组合：ecg_clean + ecg_peaks(Pan-Tompkins)
    try:
        _, info = nk.ecg_process(sig, sampling_rate=fs, method="pantompkins1985")
    except Exception as e:
        print(f"[warn] neurokit.ecg_process 失败：{e}. 尝试 clean+peaks 回退。")
        sig_clean = nk.ecg_clean(sig, sampling_rate=fs, method="biosppy")
        _, info = nk.ecg_peaks(sig_clean, sampling_rate=fs, method="pantompkins1985")

    rpeaks = info.get("ECG_R_Peaks", [])
    if rpeaks is None or len(rpeaks) < 3:
        print("[warn] ECG→RR：R 峰过少，放弃")
        return None

    rr_ms = np.diff(rpeaks) / fs * 1000.0
    t_rr  = t[rpeaks[1:]]
    return pd.DataFrame({"t_s": t_rr, "rr_ms": rr_ms})

def _read_device_rr(sid: str):
    p = RAW_CACHE_DIR / f"{sid}_rr.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    if not {"t_s","rr_ms"}.issubset(df.columns): return None
    return df

def _hr_1hz_from_rr(rr: pd.DataFrame, t0=None, t1=None, step=1.0):
    if rr is None or rr.empty: return None
    t = rr["t_s"].to_numpy(); hr = 60000.0/rr["rr_ms"].to_numpy()
    if t0 is None: t0 = float(t.min()); 
    if t1 is None: t1 = float(t.max())
    bins = np.arange(t0, t1+step, step)
    idx = np.digitize(t, bins)-1
    out = np.full(len(bins), np.nan)
    for i in range(len(bins)):
        h = hr[idx==i]
        if len(h): out[i] = np.nanmean(h)
    return bins, out

def _flag_rr(rr: pd.DataFrame):
    thr = float(PARAMS["rr_artifact_threshold"])
    hrmin, hrmax = PARAMS["hr_min_max_bpm"]
    rr = rr.copy()
    hr = 60000.0/rr["rr_ms"]
    flag_delta = rr["rr_ms"].pct_change().abs()>thr
    flag_delta.iloc[0]=False
    flag_hr = (hr<hrmin)|(hr>hrmax)
    rr["flagged"] = (flag_delta|flag_hr)
    return rr

def _apply_filter(all_sids: list[str], tokens: list[str]) -> list[str]:
    if not tokens:
        return sorted(all_sids)
    keep = []
    low = [t.strip().lower() for t in tokens if t and t.strip()]
    for sid in sorted(all_sids):
        sid_low = sid.lower()
        for t in low:
            # 短号：纯数字 → 匹配 "P{###}"
            if t.isdigit():
                tt = f"p{int(t):03d}"
                if tt in sid_low:
                    keep.append(sid); break
            # 直接包含：支持你写 f1y01 / P001… 等
            elif t in sid_low:
                keep.append(sid); break
    # 去重保持顺序
    seen = set(); out = []
    for s in keep:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def main():
    # 找到所有 被测id <sid>（有 rr 或 ecg 的算）
    sids = sorted({p.stem.split("_")[0] for p in RAW_CACHE_DIR.glob("*_rr.csv")}|{p.stem.split("_")[0] for p in RAW_CACHE_DIR.glob("*_ecg.*")})
    # 仅挑选用户指定的被试
    all_sids = sids
    sids = _apply_filter(all_sids, SUBJECTS_FILTER)
    if PREVIEW_LIMIT is not None:
        sids = sids[:int(PREVIEW_LIMIT)]
    if not sids:
        print(f"[info] 发现 {len(all_sids)} 个被试，但过滤后为 0。检查 SUBJECTS_FILTER={SUBJECTS_FILTER}")
        return
    print(f"[select] 总计发现 {len(all_sids)} 个；将处理 {len(sids)} 个 → {sids}")

    if not sids:
        print("[err] RAW_CACHE_DIR 下没找到 *_rr.csv 或 *_ecg.*"); return

    # 第一阶段：若没有决策表，就只做建议
    suggest_rows = []
    for sid in sids:
        rr_dev = _read_device_rr(sid)
        ecg = _read_ecg(sid)
        rr_ecg = _ecg_to_rr(ecg) if ecg is not None else None
        # 打印 采样率和数据长度
        if rr_ecg is None and ecg is not None:
            fs_dbg = float(ecg.get("fs_hz", pd.Series([np.nan])).iloc[0]) if "fs_hz" in ecg.columns else float("nan")
            print(f"[dbg] {sid}: ECG rows={len(ecg)}  fs_hz_in={fs_dbg}")

        if rr_dev is None and rr_ecg is None:
            print(f"[skip] {sid} 没有 RR 也没有 ECG"); continue

        # 1Hz HR 对齐
        t0 = None
        if rr_dev is not None: t0 = rr_dev["t_s"].min()
        if rr_ecg is not None: t0 = rr_ecg["t_s"].min() if t0 is None else max(t0, rr_ecg["t_s"].min())
        t1 = None
        if rr_dev is not None: t1 = rr_dev["t_s"].max()
        if rr_ecg is not None: t1 = rr_ecg["t_s"].max() if t1 is None else min(t1, rr_ecg["t_s"].max())
        if t0 is None or t1 is None or t1-t0<10:
            print(f"[warn] {sid} 可重叠时段太短"); continue
        bins, hr_dev = _hr_1hz_from_rr(rr_dev, t0,t1) if rr_dev is not None else (None, None)
        _,    hr_ecg = _hr_1hz_from_rr(rr_ecg, t0,t1) if rr_ecg is not None else (None, None)

        # 指标
        mae=bias=np.nan; valid_dev=valid_ecg=flag_dev=flag_ecg=np.nan
        if rr_dev is not None and rr_ecg is not None:
            mask = ~np.isnan(hr_dev) & ~np.isnan(hr_ecg)
            if mask.sum()>0:
                diff = hr_dev[mask]-hr_ecg[mask]
                mae = float(np.nanmean(np.abs(diff))); bias = float(np.nanmean(diff))
        if rr_dev is not None:
            r2 = _flag_rr(rr_dev); valid_dev = 1.0 - r2["flagged"].mean(); flag_dev = r2["flagged"].mean()
        if rr_ecg is not None:
            r2 = _flag_rr(rr_ecg); valid_ecg = 1.0 - r2["flagged"].mean(); flag_ecg = r2["flagged"].mean()

        # 建议
        suggest = "device_rr"
        if rr_ecg is None: suggest="device_rr"
        elif rr_dev is None: suggest="ecg_rr"
        else:
            # 按阈值判断
            ok = (mae<=RR_COMPARE["mae_bpm_max"] and abs(bias)<=RR_COMPARE["bias_bpm_max"])
            if not ok:
                suggest = "ecg_rr" if valid_ecg>=valid_dev else "device_rr"
            else:
                # 谁更干净用谁
                suggest = "ecg_rr" if flag_ecg < flag_dev else "device_rr"

        suggest_rows.append({
            "subject_id": sid,
            "has_device_rr": rr_dev is not None,
            "has_ecg": ecg is not None,
            "mae_bpm": round(mae,3),
            "bias_bpm": round(bias,3),
            "valid_ratio_device": round(valid_dev,3) if pd.notna(valid_dev) else "",
            "valid_ratio_ecg":    round(valid_ecg,3) if pd.notna(valid_ecg) else "",
            "flag_ratio_device":  round(flag_dev,3) if pd.notna(flag_dev) else "",
            "flag_ratio_ecg":     round(flag_ecg,3) if pd.notna(flag_ecg) else "",
            "choice_suggested": suggest
        })

        # 画对比图
        fig, ax = plt.subplots(figsize=(12,3))
        if rr_dev is not None: ax.plot(bins, hr_dev, label="HR from device RR")
        if rr_ecg is not None: ax.plot(bins, hr_ecg, label="HR from ECG→RR", linestyle="--")
        tit = f"{sid}  MAE={mae:.2f}  bias={bias:+.2f} bpm"
        ax.set_title(tit); ax.set_xlabel("time (s)"); ax.set_ylabel("bpm"); ax.legend(); ax.grid(True, alpha=.3)
        fig.tight_layout(); fig.savefig(SEL_DIR / f"preview_{sid}.png", dpi=130); plt.close(fig)

    # 无决策文件则生成建议并退出
    if not DEC_FILE.exists():
        pd.DataFrame(suggest_rows).to_csv(DEC_SUGG, index=False)
        print(f"[suggest] 已生成建议与预览，请人工确认后复制为 {DEC_FILE.name}。\n  - 建议表：{DEC_SUGG}\n  - 预览图：{SEL_DIR/'preview_<sid>.png'}")
        return

    # 第二阶段：读取决策，产“最终 RR”
    dec = pd.read_csv(DEC_FILE)
    # 若用户设置过滤被试id，只定稿这些 sid；否则定稿表中所有行
    if SUBJECTS_FILTER:
        allow = set(_apply_filter(sorted(dec["subject_id"].astype(str).unique()), SUBJECTS_FILTER))
        dec = dec[dec["subject_id"].astype(str).isin(allow)].copy()
        print(f"[finalize] 仅对这些被试生成最终 RR：{sorted(allow)}")

    out_rows=[]
    for _, r in dec.iterrows():
        sid = str(r["subject_id"]); choice = str(r["choice_suggested"]).strip().lower()
        if "choice" in dec.columns: choice = str(r["choice"]).strip().lower() or choice
        rr = _read_device_rr(sid) if choice=="device_rr" else None
        if rr is None:
            ecg = _read_ecg(sid)
            rr = _ecg_to_rr(ecg) if ecg is not None else None
        if rr is None:
            print(f"[warn] {sid} 无法生成最终 RR"); continue
        out = PROCESSED_DIR / f"2rr_{sid}.csv"
        rr.to_csv(out, index=False)
        print(f"[2rr] {sid} → {out.name} (rows={len(rr)})")
        out_rows.append({"subject_id":sid, "rows":len(rr), "source":choice})
    if out_rows:
        pd.DataFrame(out_rows).to_csv(SEL_DIR/"final_rr_summary.csv", index=False)
        print(f"[save] 最终 RR 概览 → {SEL_DIR/'final_rr_summary.csv'}")
    print("[done] RR 选择完成。")
if __name__=="__main__":
    main()