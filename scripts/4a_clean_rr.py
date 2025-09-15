# scripts/2_clean_rr.py
# 功能：ECG → R峰 → RR；按可解释规则标注伪迹，输出清洗后RR表（parquet+csv），
#      打印清洗规则与汇总、总进度条；导出“伪迹复核”csv以便人工二次确认，可选回灌应用。
# ----------------------------
# 清理规则：
# 1.前后两次心跳差太大就判可疑。
# 计算心跳变化幅度：“相对变化幅度”= |本次 RR − 上次 RR| ÷ 上次 RR。
# 当前设置为0.20，也就是超过 20% 就判“可疑”。
# 2.心率不符合常理就判可疑
# 把 RR 换算成心率 = 60000 ÷ RR。我们用的范围是 35 到 140 次/分
# 心率跳到 180 或掉到 25，不像生理波动
# ⚠️：变化幅度与心跳阈值 在settings的PARAMS里修改
# ----------------------------
# 人工复核机制：
# 脚本运行后，会在 data/processed/review下生成名为 rr_flags_<sid>.csv 的文件
# 里有 keep 和 rr_ms_override 两列可手工填写
# keep=1 强制把该行设为 valid=True
# rr_ms_override 若填正数，会替换该行 rr_ms 并重算 hr_bpm
# 启动复核机制，先填写review表格，再把脚本顶部 REVIEW_APPLY=True，再次运行脚本
# 默认关闭回灌，避免误操作

import pandas as pd
import numpy as np
import neurokit2 as nk

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

from settings import PROCESSED_DIR, RAW_CACHE_DIR, PARAMS

# 可选：仅挑选部分被试（留空或 None 表示不过滤）
SUBJECTS_FILTER = ["f1y01", "f1o01"]

# 复核导出与回灌（回灌默认关，避免误操作）
REVIEW_EXPORT = True
REVIEW_APPLY = False   # True 时，会读取 review/rr_flags_<sid>.csv 中的人工“keep”标记回灌

REVIEW_DIR = PROCESSED_DIR / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

# 进度条（没有 tqdm 也能跑）
try:
    from tqdm import tqdm
    def track(it, total, desc): return tqdm(it, total=total, desc=desc)
except Exception:
    def track(it, total, desc):
        print(f"[progress] {desc} total={total}")
        for i, x in enumerate(it, 1):
            if i == 1 or i == total or i % max(1, total//10) == 0:
                print(f"[progress] {desc} {i}/{total}")
            yield x

def _subjects_from_cache() -> list[str]:
    return sorted({p.stem.replace("_ecg","") for p in (RAW_CACHE_DIR.glob("*_ecg.parquet"))})

def _clean_rr_from_ecg_series(time_s: np.ndarray, ecg: np.ndarray, fs: float) -> pd.DataFrame:
    """
    返回列：
      t_s, rr_ms, hr_bpm, valid, flagged, flag_delta, flag_hr, flag_reason, n_rr_corrections
    说明：
      - flagged 表示“按原始规则被标记为伪迹”
      - valid 表示“用于下游分析是否有效”：delete 策略下 flagged->False；interp 下 flagged 但 valid=True
    """
    # R峰检测
    _, info = nk.ecg_process(ecg, sampling_rate=fs)
    rpeaks = info.get("ECG_R_Peaks", None)
    if rpeaks is None or len(rpeaks) < 3:
        return pd.DataFrame(columns=[
            "t_s","rr_ms","hr_bpm","valid","flagged","flag_delta","flag_hr","flag_reason","n_rr_corrections"
        ])

    # 原始 RR 与 HR
    rr_raw = np.diff(rpeaks) / fs * 1000.0  # ms
    t_rr    = time_s[rpeaks[1:]]
    hr_raw  = 60000.0 / rr_raw

    # 规则：相邻RR相对变化>阈值，或 HR 越界
    thr = float(PARAMS["rr_artifact_threshold"])
    hr_min, hr_max = PARAMS["hr_min_max_bpm"]
    flag_delta = np.abs(pd.Series(rr_raw).pct_change()) > thr
    flag_delta.iloc[0] = False  # 第一跳没有 previous
    flag_hr = (hr_raw < hr_min) | (hr_raw > hr_max)

    flagged = (flag_delta | flag_hr).to_numpy()
    flag_reason = np.where(flagged, np.where(flag_delta & flag_hr, "delta|hr",
                                    np.where(flag_delta, "delta", "hr")), "")

    # 纠正策略
    rr_clean = rr_raw.copy()
    if PARAMS["rr_fix_strategy"] == "interp" and flagged.any():
        idx_bad = np.where(flagged)[0]
        idx_ok  = np.where(~flagged)[0]
        if len(idx_ok) >= 2:
            rr_clean[idx_bad] = np.interp(idx_bad, idx_ok, rr_clean[idx_ok])

    hr_clean = 60000.0 / rr_clean

    # valid 定义
    if PARAMS["rr_fix_strategy"] == "delete":
        valid = ~flagged
    else:  # interp
        valid = np.ones_like(flagged, dtype=bool)

    n_corr = np.cumsum(flagged)

    rr_df = pd.DataFrame({
        "t_s": t_rr.astype(float),
        "rr_ms": rr_clean.astype(float),
        "hr_bpm": hr_clean.astype(float),
        "valid": valid,
        "flagged": flagged,
        "flag_delta": flag_delta.to_numpy(),
        "flag_hr": flag_hr.astype(bool),
        "flag_reason": flag_reason,
        "n_rr_corrections": n_corr.astype(int),
    })
    return rr_df

def _apply_review_if_available(sid: str, rr_df: pd.DataFrame) -> pd.DataFrame:
    """
    回灌人工复核：
      - 如果 review/rr_flags_<sid>.csv 存在且含 'keep' 列：
           keep==1 → 强制设为 valid=True
           keep==0 → 保持当前 valid（delete 下为 False；interp 下仍 True，但 flagged 标记保留）
      - 若含 rr_ms_override 列且是正数 → 用该值替换 rr_ms，并重算 hr_bpm
    """
    review_file = REVIEW_DIR / f"rr_flags_{sid}.csv"
    if not REVIEW_APPLY or not review_file.exists():
        return rr_df

    try:
        rev = pd.read_csv(review_file)
    except Exception as e:
        print(f"[review] {sid}: cannot read {review_file}: {e}")
        return rr_df

    if "t_s" not in rev.columns:
        print(f"[review] {sid}: {review_file} missing 't_s' column; skip apply.")
        return rr_df

    merged = rr_df.merge(rev[["t_s","keep","rr_ms_override"]].fillna({'keep':np.nan,'rr_ms_override':np.nan}),
                         on="t_s", how="left")

    # 应用 keep
    if "keep" in merged.columns:
        enforce_keep = merged["keep"] == 1
        merged.loc[enforce_keep, "valid"] = True

    # 应用 rr_ms_override
    if "rr_ms_override" in merged.columns:
        mask = pd.to_numeric(merged["rr_ms_override"], errors="coerce") > 0
        merged.loc[mask, "rr_ms"] = merged.loc[mask, "rr_ms_override"]
        merged["hr_bpm"] = 60000.0 / merged["rr_ms"]

    merged.drop(columns=[c for c in ["keep","rr_ms_override"] if c in merged.columns], inplace=True)
    return merged

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    subjects = _subjects_from_cache()
    if SUBJECTS_FILTER:
        subjects = [s for s in subjects if s in SUBJECTS_FILTER]

    # 规则透明化：把清洗口径打印出来
    print("[rules] RR 伪迹判定：")
    print(f"        1) 相邻RR相对变化 > {PARAMS['rr_artifact_threshold']:.2f} 记为 delta")
    print(f"        2) 心率越界 (<{PARAMS['hr_min_max_bpm'][0]} 或 >{PARAMS['hr_min_max_bpm'][1]}) 记为 hr")
    print(f"        修正策略：{PARAMS['rr_fix_strategy']} （'delete' 删除；'interp' 插值替换）")
    print(f"        输出：parquet + csv；复核文件：{REVIEW_EXPORT}, 回灌：{REVIEW_APPLY}\n")

    summ_rows = []
    for sid in track(subjects, total=len(subjects), desc="RR cleaning"):
        # 读每被试缓存
        ecg_df = pd.read_parquet(RAW_CACHE_DIR / f"{sid}_ecg.parquet")
        fs = float(ecg_df["fs_hz"].iloc[0])
        time_s = ecg_df["time_s"].to_numpy()
        ecg = ecg_df["value"].to_numpy()

        rr_df = _clean_rr_from_ecg_series(time_s, ecg, fs)

        # 导出复核文件（仅含被标记的行，并附规则字段）
        if REVIEW_EXPORT and len(rr_df):
            flags = rr_df[rr_df["flagged"]].copy()
            if not flags.empty:
                flags["hr_bpm_raw_minmax"] = f"{PARAMS['hr_min_max_bpm'][0]}–{PARAMS['hr_min_max_bpm'][1]}"
                flags["rule_delta_thr"] = float(PARAMS["rr_artifact_threshold"])
                # 给人工复核的两列：keep（0/1），rr_ms_override（可选）
                flags["keep"] = np.nan
                flags["rr_ms_override"] = np.nan
                flags_out = REVIEW_DIR / f"rr_flags_{sid}.csv"
                # 只留最有用的列，便于人工查看
                cols = ["t_s","rr_ms","hr_bpm","flag_delta","flag_hr","flag_reason",
                        "rule_delta_thr","hr_bpm_raw_minmax","keep","rr_ms_override"]
                flags[cols].to_csv(flags_out, index=False)
                print(f"[review] {sid}: flagged={len(flags)} → {flags_out}")

        # 回灌人工复核（可选）
        rr_df = _apply_review_if_available(sid, rr_df)

        # 汇总统计
        n_total = len(rr_df)
        n_flag  = int(rr_df["flagged"].sum())
        n_valid = int(rr_df["valid"].sum())
        n_delta = int(rr_df["flag_delta"].sum())
        n_hr    = int(rr_df["flag_hr"].sum())
        pct_flag = (n_flag / n_total * 100.0) if n_total else 0.0
        pct_valid = (n_valid / n_total * 100.0) if n_total else 0.0
        summ_rows.append({
            "subject_id": sid,
            "n_rr": n_total,
            "n_flagged": n_flag,
            "pct_flagged": round(pct_flag, 2),
            "n_flag_delta": n_delta,
            "n_flag_hr": n_hr,
            "n_valid": n_valid,
            "pct_valid": round(pct_valid, 2),
            "strategy": PARAMS["rr_fix_strategy"],
        })

        # 输出：parquet + csv（便于快速浏览）
        out_parquet = PROCESSED_DIR / f"2clean_{sid}.parquet"
        out_csv     = PROCESSED_DIR / f"2clean_{sid}.csv"
        rr_df.to_parquet(out_parquet, index=False)
        rr_df.to_csv(out_csv, index=False)
        print(f"[2] {sid}: RR cleaned → {out_parquet.name} & {out_csv.name} "
              f"(rows={n_total}, valid={n_valid}, flagged={n_flag})")

    # 写入汇总表
    if summ_rows:
        summary_df = pd.DataFrame(summ_rows)
        summary_out = PROCESSED_DIR / "2clean_summary.csv"
        summary_df.to_csv(summary_out, index=False)
        print(f"[2] summary saved → {summary_out}")
        # 控制台也给一眼看懂的 Top-5
        print(summary_df.head().to_string(index=False))

if __name__ == "__main__":
    main()