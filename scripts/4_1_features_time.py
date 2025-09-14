import pandas as pd
import numpy as np
from settings import PROCESSED_DIR

def rmssd_ms(rr_ms):
    diff = np.diff(rr_ms)
    return np.sqrt(np.mean(diff**2)) if len(diff)>0 else np.nan

def sdnn_ms(rr_ms):
    return float(np.std(rr_ms, ddof=1)) if len(rr_ms)>1 else np.nan

def pnn50_pct(rr_ms):
    diff = np.abs(np.diff(rr_ms))
    return 100.0 * (diff > 50.0).mean() if len(diff)>0 else np.nan

def mean_hr_bpm(rr_ms):
    return 60000.0 / np.mean(rr_ms) if len(rr_ms)>0 else np.nan

def poincare_sd1_sd2(rr_ms):
    if len(rr_ms) < 2: return (np.nan, np.nan)
    diff = np.diff(rr_ms)
    sd1 = np.sqrt(np.var(diff)/2.0)
    sd2 = np.sqrt(2*np.var(rr_ms) - (np.var(diff)/2.0))
    return (float(sd1), float(sd2))

def main():
    for p in PROCESSED_DIR.glob("3win_*.csv"):
        sid = p.stem.replace("3win_","")
        win = pd.read_csv(p)
        rr = pd.read_parquet(PROCESSED_DIR / f"2clean_{sid}.parquet")
        rows = []
        for _, w in win.iterrows():
            seg = rr[(rr["t_s"]>=w.window_start_s) & (rr["t_s"]<w.window_start_s + w.window_dur_s)]
            seg = seg[seg["valid"]==True]
            rrm = seg["rr_ms"].to_numpy()
            rows.append({
                "subject_id": sid,
                "window_id": w.window_id,
                "mean_hr_bpm": mean_hr_bpm(rrm),
                "rmssd_ms": rmssd_ms(rrm),
                "sdnn_ms": sdnn_ms(rrm),
                "pnn50_pct": pnn50_pct(rrm),
                "sd1_ms": poincare_sd1_sd2(rrm)[0],
                "sd2_ms": poincare_sd1_sd2(rrm)[1],
            })
        out = pd.DataFrame(rows)
        out.to_csv(PROCESSED_DIR / f"4feat_time_{sid}.csv", index=False)
        print(f"[4] time features saved for {sid}")

if __name__ == "__main__":
    main()