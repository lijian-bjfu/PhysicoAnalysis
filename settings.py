# 需要安装依赖：pip install wfdb pandas numpy scipy neurokit2 matplotlib

from pathlib import Path
# 数据源开关
DATA_SOURCE = "fantasia"   # "local" or "fantasia"

# 数据根
PROJECT_ROOT  = Path(__file__).parent
DATA_DIR      = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_CACHE_DIR = DATA_DIR / "raw" / ("fantasia" if DATA_SOURCE=="fantasia" else "local")

# 指定数据集
ACTIVE_DATA = "local"

DATASETS = {
    "local": {
        "loader": "scripts.loaders.custom_loader",   # 只负责“归位”
        # "root":   "raw/local",                        # 统一落地目录,该目录接 DATA_DIR
        "events": "data/raw/local/events",                 # 事件落地目录
        "options": {
            "sid_pattern": "P{p:03d}S{s:03d}T{t:03d}R{r:03d}",  # 文件名生成用
            "ask_dir": True,                                   # 打开系统对话框选目录
            "copy_mode": "copy2"                               # copy/copy2/move
        },
        "paths": {
            "raw":       "raw/local",                  # 归位后的原始（或导入）文件
            "norm":      "processed/norm/local",       # 2_data_norm 的输出
            "confirmed": "processed/confirmed/local",  # 3_select_rr 最终 RR
            "features":  "processed/features/local",   # 6x 特征输出
            "events":    "raw/local/events"            # 原始事件标记（若有）
        }
    },
    "fantasia": {
        "loader": "scripts.loaders.fantasia_loader",
        "root":   "raw/fantasia",
        "events": None,
        "options": {
            "records": [],                   # 空=对所有被试数据处理；也可列 ["f1o05","f1o06",...]
            "signals": ["ecg","resp"],       # 要落盘的信号
            "prefer_local_wfdb": True,       # 有 .hea/.dat/.ecg 就本地解析
            "allow_network": False,          # 禁止联网（你已有 subset）
            "cache_format": "parquet"        # 产 <sid>_ecg.parquet / <sid>_resp.parquet
        },
        "paths": {
            "raw":       "raw/fantasia/",
            "norm":      "processed/norm/fantasia",
            "confirmed": "processed/confirmed/fantasia",
            "features":  "processed/features/fantasia",
        }
    }
    # "wesad": {...} 以后你加
}

def P(kind: str, ds: str | None = None) -> Path:
    """解析 dataset 的某类路径（raw/norm/events/rr_final/features）。"""
    ds = ds or ACTIVE_DATA
    rel = DATASETS[ds]["paths"][kind]
    return (DATA_DIR / rel).resolve()

# 兼容旧代码的别名（尽量让老脚本不炸）
RAW_CACHE_DIR = P("raw")
EVENTS_DIR    = P("events") if "events" in DATASETS[ACTIVE_DATA]["paths"] else (DATA_DIR / "raw/events")


# 导入器关键词（给 1_data_norm.py 用）
# 信号别名（用于 detect_signal）
SIGNAL_ALIASES = {
    "ecg":   ["ecg"],
    "rr":    ["rr","ibi","ppi"],
    "hr":    ["hr","bpm","heart_rate"],
    "resp":  ["resp","respiration","breath","breathing"],
    "ppg":   ["ppg","bvp"],
    "acc":   ["acc","accelerometer","accel"],
    # "events":["events","event","marker","mark","trigger","onset"]
}
DEVICE_TAGS = {"h10":"h", "verity":"v"}


# RR 选择阈值（给 2a_select_rr.py）
RR_COMPARE = {
    "hr_smooth_sec": 1.0,     # 1 Hz 心率轨
    "mae_bpm_max": 2.0,
    "bias_bpm_max": 1.0,
    "min_valid_ratio": 0.95,
    "max_flag_ratio": 0.05,
    "max_drift_ms_per5min": 200.0
}

# RR 综合评分权重（给 3_select_rr.py）
# 说明：
# - 评分目标是“越大越好”： score = w_flag*(1-flag_ratio) + w_mae*(1 - min(MAE/mae_bpm_max,1)) + w_bias*(1 - min(|bias|/bias_bpm_max,1))
# - 权重可不必严格相加为 1，但建议 w_flag + w_mae + w_bias ≈ 1
# - 如需更看重一致性（MAE/Bias），可调大 w_mae / w_bias；更看重干净程度，则调大 w_flag
RR_SCORE = {
    "w_flag": 0.50,   # 伪迹率越低越好（权重）
    "w_mae":  0.30,   # 与另一来源的一致性（1Hz HR 的 MAE）
    "w_bias": 0.20,   # 系统性偏差（绝对 Bias）
}

# 清洗与分窗参数（全局）
PARAMS = {
    "rr_artifact_threshold": 0.20,   # 相邻RR变化率>20% → 伪迹，非静息时，阈值放宽
    "hr_min_max_bpm": (35, 140),
    "rr_fix_strategy": "delete",     # 'delete' or 'interp'
    "window_sec": 300,               # 标准5分钟
    "overlap_sec": 0,
    "simulate_T0_T3": False,         # 若True: 5/4/5/5/5/5分的伪T0–T3
    "hf_band": (0.15, 0.40),
    "use_individual_hf": True,
    "hf_band_radius_hz": 0.05,
    "log_power": True,
    "valid_rr_ratio_min": 0.80,
    "require_5min_for_freq": True,
}

# 原始数据快查表 2qc_rr_<sid>*.csv
QC_RR_LABELS = {
    "t_s": "RR 对应的时间戳（秒，ECG 时间轴）",
    "rr_ms": "心搏间期（毫秒）",
    "hr_bpm": "由 rr_ms 换算的心率（次/分）",
    "valid": "该 RR 是否被判为有效（布尔）；等于 not flagged",
    "flagged": "点级规则命中（布尔）；相邻 RR 相对变化超阈值或心率越界即为 True",
    "flag_delta": "相邻 RR 相对变化是否超过阈值 rr_artifact_threshold（布尔）",
    "flag_hr": "心率是否越过 hr_min_max_bpm 范围（布尔）",
    "flag_reason": "命中规则的摘要：'delta'、'hr' 或 'delta|hr'；未命中为空字符串",
    "quality_check": "快速质检用的宽口径可疑标记（布尔）；推荐定义为 flagged 或位于建议排查区间内",
    "n_rr_corrections": "从序列开始累计到当前点，被判为可疑/需修正的 RR 个数（仅计数，未实际改动）"
}

# 清洗后单被试 RR 表（2clean_<sid>.csv / .parquet）
RR_CLEAN_LABELS = {
    "t_s": "本次心搏的对齐时间（秒，基于原始ECG时间轴中R峰的第二个及以后峰值）",
    "rr_ms": "相邻心搏间期（毫秒），应用修正策略后的最终数值（delete不改值、interp为插值）",
    "hr_bpm": "由 rr_ms 换算的心率（每分钟心跳数），等于 60000 / rr_ms",
    "valid": "是否用于下游分析的有效标记；在 delete 策略下等于非伪迹，在 interp 策略下一律为 True",
    "flagged": "是否被任何伪迹规则标记（True/False），仅用于透明化，不等同于是否纳入分析",
    "flag_delta": "是否因“相邻RR相对变化超过阈值”被标记（True/False），阈值见 PARAMS['rr_artifact_threshold']",
    "flag_hr": "是否因“心率越界”被标记（True/False），上下界见 PARAMS['hr_min_max_bpm']",
    "flag_reason": "伪迹原因字符串：'delta'、'hr'、'delta|hr' 或空字符串",
    "n_rr_corrections": "从序列开始累计被标记次数的计数（单调递增），用于快速定位伪迹密集区"
}

# 清洗汇总表（2clean_summary.csv）
RR_CLEAN_SUMMARY_LABELS = {
    "subject_id": "被试编号（与原始记录名一致）",
    "n_rr": "该被试的RR条目总数（心搏对数）",
    "n_flagged": "被伪迹规则标记的RR条目数（flagged为True的计数）",
    "pct_flagged": "被标记条目的百分比（100 * n_flagged / n_rr，保留两位小数）",
    "n_flag_delta": "因delta规则（相邻RR相对变动超阈）被标记的条目数",
    "n_flag_hr": "因hr规则（心率越界）被标记的条目数",
    "n_valid": "最终用于分析的RR条目数（delete：n_rr - n_flagged；interp：等于n_rr）",
    "pct_valid": "有效RR占比（100 * n_valid / n_rr）",
    "strategy": "采用的修正策略（'delete' 删除伪迹行；'interp' 插值覆盖伪迹行）"
}

# RR 伪迹人工复查表（data/processed/review/rr_flags_<sid>.csv）
RR_REVIEW_LABELS = {
    "t_s": "心搏时间戳（秒），与清洗后RR表可一一对齐",
    "rr_ms": "当前RR（毫秒），在delete策略下为原始RR，在interp策略下为已插值值",
    "hr_bpm": "由rr_ms换算的心率（每分钟）",
    "flag_delta": "该条是否因delta规则被标记",
    "flag_hr": "该条是否因hr规则被标记",
    "flag_reason": "伪迹原因字符串：'delta'、'hr' 或 'delta|hr'",
    "rule_delta_thr": "delta规则使用的相邻RR相对变化阈值（来自PARAMS['rr_artifact_threshold']）",
    "hr_bpm_raw_minmax": "心率越界判定使用的上下界（来自PARAMS['hr_min_max_bpm']）",
    "keep": "人工复核保留标记（留空=不改变；1=强制设为有效）",
    "rr_ms_override": "人工给出的RR替代值（毫秒，正数才生效），用于手动修正异常RR"
}

# 建议伪迹清楚表
QC_SUGGEST_LABELS = {
    "subject_id": "被试编号，与 2clean_<sid>.csv 对应",
    "t_start_s": "建议剪除片段起点（秒，ECG 时间轴）",
    "t_end_s": "建议剪除片段终点（秒，ECG 时间轴）",
    "reason": "建议原因标签的并集（low_valid/high_flag/hr_jump/too_few_rr 等）"
}

# 可调质检参数默认值
QC_PARAMS = {
    "qc_window_s": 30.0,        # 质检滑窗长度（秒）
    "qc_stride_s": 10.0,        # 质检滑窗步长（秒）
    "qc_min_rr_per_win": 20,    # 每窗最少 RR 数
    "qc_min_valid_ratio": 0.85, # 有效 RR 比例阈值（低于此为差）
    "qc_max_flagged_ratio": 0.20,# 被标记 RR 比例阈值（高于此为差）
    "qc_hr_jump_bpm": 25.0,     # 相邻窗 HR 均值跳变阈值（bpm）
    "qc_merge_gap_s": 5.0,      # 建议片段间隙合并阈值（秒）
    "qc_min_cut_s": 3.0,        # 最短建议片段长度（秒）
}

# 输出列说明
OUTPUT_LABELS = {
    "subject_id": "被试唯一标识（字符串，例如 'sub-0001'）",
    "age_group": "年龄组（young / old / unknown）",
    "window_id": "时间窗口编号（如 'win-001' 或 'T2b'）",
    "window_start_s": "窗口起始时间（秒，记录起点=0）",
    "window_dur_s": "窗口时长（秒；5分钟=300）",
    "valid_rr_ratio": "有效RR占比（0–1；≥0.80 推荐有效）",
    "n_rr_corrections": "RR校正次数（删除/插值的心搏数量）",
    "window_valid": "窗口有效标记（布尔）",
    "resp_rate_bpm": "呼吸频率（次/分；缺失则留空）",
    "hf_band_used": "HF频带类型（fixed/individual）",
    "hf_center_hz": "个体化HF中心频率（Hz）",
    "mean_hr_bpm": "平均心率（次/分）",
    "rmssd_ms": "RMSSD（毫秒）",
    "sdnn_ms": "SDNN（毫秒）",
    "pnn50_pct": "pNN50（百分比）",
    "sd1_ms": "SD1（毫秒）",
    "sd2_ms": "SD2（毫秒）",
    "hf_log_ms2": "HF功率自然对数 ln(ms²)",
    "lf_log_ms2": "LF功率自然对数 ln(ms²)",
    "rsa_ms": "RSA（毫秒；可缺）",
}