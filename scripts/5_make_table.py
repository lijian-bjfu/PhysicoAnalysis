import pandas as pd
from settings import PROCESSED_DIR, FEATURES_DIR

def main():
    feats = []
    # 合并时间域
    time_parts = list(PROCESSED_DIR.glob("4feat_time_*.csv"))
    for tp in time_parts:
        feats.append(pd.read_csv(tp))
    time_df = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()

    # 合并频域
    freq_parts = list(PROCESSED_DIR.glob("5feat_freq_*.csv"))
    freq_df = pd.concat([pd.read_csv(x) for x in freq_parts], ignore_index=True) if freq_parts else pd.DataFrame()

    # 合并窗口质量
    win_parts = list(PROCESSED_DIR.glob("3win_*.csv"))
    win_df = pd.concat([pd.read_csv(x) for x in win_parts], ignore_index=True) if win_parts else pd.DataFrame()

    df = time_df.merge(freq_df, on=["subject_id","window_id"], how="left").merge(
        win_df[["subject_id","window_id","window_start_s","window_dur_s","valid_rr_ratio"]],
        on=["subject_id","window_id"], how="left"
    )
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_DIR / "fantasia_features_long.csv", index=False)
    print(f"[7] merged table saved: {FEATURES_DIR / 'fantasia_features_long.csv'}")

if __name__ == "__main__":
    main()