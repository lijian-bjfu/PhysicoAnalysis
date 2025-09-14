import pandas as pd
# --- project-root bootstrap ---
import sys
from pathlib import Path
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent))
        break
    _p = _p.parent
# --- end bootstrap ---
from settings import PROCESSED_DIR, PARAMS

def _make_windows(start_s: float, end_s: float, win: int, step: int):
    t = start_s
    idx = 1
    while t + win <= end_s + 1e-6:
        yield (f"win-{idx:03d}", t, win)
        t += step
        idx += 1

def main():
    # 读一个被试的 RR（示例也可批量）
    # 这里演示：遍历 2clean_* 文件
    for p in PROCESSED_DIR.glob("2clean_*.parquet"):
        sid = p.stem.replace("2clean_","")
        rr = pd.read_parquet(p)
        start, end = float(rr["t_s"].min()), float(rr["t_s"].max())
        win = int(PARAMS["window_sec"]); step = win - int(PARAMS["overlap_sec"])
        rows = []
        for wid, ws, dur in _make_windows(start, end, win, step):
            seg = rr[(rr["t_s"]>=ws) & (rr["t_s"]<ws+dur)]
            valid_rr_ratio = seg["valid"].mean() if len(seg)>0 else 0.0
            rows.append({"subject_id": sid, "window_id": wid, "window_start_s": ws, "window_dur_s": dur,
                         "valid_rr_ratio": valid_rr_ratio})
        win_df = pd.DataFrame(rows)
        win_df.to_csv(PROCESSED_DIR / f"3win_{sid}.csv", index=False)
        print(f"[3] windows saved for {sid}")

if __name__ == "__main__":
    main()