import pandas as pd
import numpy as np
from settings import PROCESSED_DIR, PARAMS
from scipy.signal import welch

def hrv_psd(rr_ms, fs_interp=4.0):
    # RR插值到等间隔心率时间序列（简单线性），再做Welch
    if len(rr_ms) < 3: return None
    # 构造“心搏时刻”序列
    t = np.cumsum(rr_ms)/1000.0
    t = t - t[0]
    # 插值为等间隔
    t_uniform = np.arange(0, t[-1], 1.0/fs_interp)
    rr_interp = np.interp(t_uniform, t, rr_ms)
    # 去趋势
    rr_detr = rr_interp - np.mean(rr_interp)
    f, pxx = welch(rr_detr, fs=fs_interp, nperseg=min(len(rr_detr), 256))
    return f, pxx

def band_power(f, pxx, lo, hi):
    mask = (f>=lo) & (f<=hi)
    return np.trapz(pxx[mask], f[mask])

def main():
    for p in PROCESSED_DIR.glob("3win_*.csv"):
        sid = p.stem.replace("3win_","")
        win = pd.read_csv(p)
        rr = pd.read_parquet(PROCESSED_DIR / f"2clean_{sid}.parquet")
        rows = []
        for _, w in win.iterrows():
            if PARAMS["require_5min_for_freq"] and abs(w.window_dur_s - 300) > 1:
                rows.append({"subject_id":sid,"window_id":w.window_id,"hf_log_ms2":np.nan,"lf_log_ms2":np.nan,
                             "hf_band_used":"fixed","hf_center_hz":np.nan})
                continue
            seg = rr[(rr["t_s"]>=w.window_start_s) & (rr["t_s"]<w.window_start_s + w.window_dur_s)]
            seg = seg[seg["valid"]==True]
            rrm = seg["rr_ms"].to_numpy()
            if len(rrm) < 3:
                rows.append({"subject_id":sid,"window_id":w.window_id,"hf_log_ms2":np.nan,"lf_log_ms2":np.nan,
                             "hf_band_used":"fixed","hf_center_hz":np.nan})
                continue
            f, pxx = hrv_psd(rrm)
            if f is None:
                rows.append({"subject_id":sid,"window_id":w.window_id,"hf_log_ms2":np.nan,"lf_log_ms2":np.nan,
                             "hf_band_used":"fixed","hf_center_hz":np.nan})
                continue
            hf_lo, hf_hi = PARAMS["hf_band"]
            band_used = "fixed"
            hf_center = np.nan
            # TODO: 若你有呼吸频率，可将 hf_lo/hi = resp_peak ± radius
            hf = band_power(f, pxx, hf_lo, hf_hi)
            lf = band_power(f, pxx, 0.04, 0.15)
            hf_log = np.log(hf) if PARAMS["log_power"] and hf>0 else np.nan
            lf_log = np.log(lf) if PARAMS["log_power"] and lf>0 else np.nan
            rows.append({"subject_id":sid,"window_id":w.window_id,"hf_log_ms2":hf_log,"lf_log_ms2":lf_log,
                         "hf_band_used":band_used,"hf_center_hz":hf_center})
        out = pd.DataFrame(rows)
        out.to_csv(PROCESSED_DIR / f"5feat_freq_{sid}.csv", index=False)
        print(f"[5] freq features saved for {sid}")

if __name__ == "__main__":
    main()