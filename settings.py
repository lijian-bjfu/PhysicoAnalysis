# 需要安装依赖：pip install wfdb pandas numpy scipy neurokit2 matplotlib

from pathlib import Path
# 数据源开关
DATA_SOURCE = "fantasia"   # "local" or "fantasia"

# 数据根
PROJECT_ROOT  = Path(__file__).parent
DATA_DIR      = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_CACHE_DIR = DATA_DIR / "raw" / ("fantasia" if DATA_SOURCE=="fantasia" else "local")

# 全局参数，包括清理、切窗、特征提取等数据处理需要的参数
PARAMS = {
    # 相邻RR变化率>20% → 伪迹，非静息时，阈值放宽
    "rr_artifact_threshold": 0.20,   
    "hr_min_max_bpm": (35, 140),
    # 'delete' or 'interp'
    "rr_fix_strategy": "delete",     
    # 切窗长度，标准5分钟
    "window_sec": 300,  
    "min_window": 1,             
    "overlap_sec": 0,
    "valid_rr_ratio_min": 0.80,
    "require_5min_for_freq": True,

    # —— 时域口径 （供 hrv_time 使用） ——
    # pNNx 阈值（ms），默认 pNN50
    "pnn_threshold_ms": 50.0,   
    # 1=样本标准差（推荐），0=总体标准差
    "sdnn_ddof": 1,             

    # —— 频域口径（供 hrv_freq 使用）——
    # 若True: 5/4/5/5/5/5分的伪T0–T3
    "simulate_T0_T3": False,         # 若True: 5/4/5/5/5/5分的伪T0–T3
    # 将不等间隔 RR 插值为等间隔心动间期序列的内部频谱采样率,默认 4 Hz
    "fs_interp": 4,
    # HF 固定带宽 (lo, hi) Hz
    "hf_band": (0.15, 0.40),  
    # 计算 lf 时，如果有呼吸数据，可做个性化 hf, 打开此参数，个体化 HF基于呼吸峰±半径计算
    "use_individual_hf": True, 
    # 结合呼吸的个性化 hf 的半径（Hz），0.05 表示呼吸峰±0.05。此参数用于调整带宽的跨度。
    "hf_band_radius_hz": 0.05,
    # 是否对 hrv lf, hf 进行对数化操作（ln ms²），默认打开
    "log_power": True,

    # —— RSA 窦性呼吸性心律不齐口径（供 hrv_rsa 使用） ——
    # RSA 前是否把该段呼吸时间重置为0，只影响 RSA 模块
    "rsa_rebase_resp": True,
    # 呼吸生理范围（次/分）。用于筛掉不合生理的峰间期：
    #   最低 6 次/分 ≈ 0.10 Hz；最高 30 次/分 ≈ 0.50 Hz。
    #   若你的被试常进行慢呼吸训练，可适当下调 resp_min_bpm（例如 4）。
    "resp_min_bpm": 6.0,
    "resp_max_bpm": 30.0,

    # 呼吸峰检测的“显著性”门槛（scipy.signal.find_peaks 的 prominence）。
    #   0 表示不过滤，仅凭距离与生理范围约束。数值越大，挑选的峰越“尖锐”，可抑制噪声导致的虚假峰。
    #   经验：干净带子=0；PPG/胸带偶有飘动=0.1～0.3（需按数据幅度调整）。
    "resp_peak_prominence": 0.0,

    # 每个呼吸周期内用于计算 RSA（max RR - min RR）的最少 RR 个数。
    #   若 RR 太稀/段太短，周期内 RR 少于该阈值将被跳过，避免极端值。
    "rsa_min_rr_per_breath": 2,

    # 对多个呼吸周期的 RSA 汇总方式：'mean' 或 'median'。
    #   一般用 'mean'；若存在少数异常大/小的周期，'median' 更稳健。
    "rsa_agg": "mean",

    # （可选）呼吸采样率覆盖值（Hz）。
    #   若你的呼吸设备恒定采样率且文件中时间戳不可靠，可在此指定固定采样率，
    #   例如 50.62851860274504。留空或 None 表示不覆盖，仍以数据时间戳为准。
    "resp_fs_hz_override": None,

    # （可选）将呼吸视为“等间隔采样”标志。
    #   True：在部分工具链只提供样点序号的场景下，可结合 resp_fs_hz_override 生成时间轴；
    #   False：严格使用文件中提供的时间戳（推荐）。
    "resp_assume_uniform": False,
    
}

# 指定数据集
ACTIVE_DATA = "fantasia"

DATASETS = {
    "local": {
        "loader": "scripts.loaders.local_loader",   # 只负责“归位”
        "options": {
            "sid_pattern": "P{p:03d}S{s:03d}T{t:03d}R{r:03d}",  # 文件名生成用
            "ask_dir": True,                                   # 打开系统对话框选目录
            "copy_mode": "copy2",                              # 把原始文件载入 raw 文件夹的方法：copy/copy2/move
        },
        "paths": {
            "raw":       "raw/local",                  # 归位后的原始（或导入）文件
            "groups":   "groups/local",                 # 分组信息
            "norm":      "processed/norm/local",       # 2_data_norm 的输出
            "confirmed": "processed/confirmed/local",  # 3_select_rr 最终 RR
            "clean":        "processed/clean/local",   # 检验并清理数据
            "preview":      "processed/preview/local",
            "windowing": "processed/windowing/local",  # 切窗
            "features":  "processed/features/local",   # 6x 特征输出
            "final":    "processed/final/local",        # 最终的数据表
            "events":    "raw/local/events"            # 原始事件标记（若有）
        },
        # 分组信息。分组信息多数可以从sid中获取
        "groups": {
            "sample": "P001S001T001R001",
            # 切片法，指定哪几个字符是分组信息
            "group_map": {
                "session": {"start": 4, "end": 7},  # 左闭右开，0 起算
                "task":    {"start": 8, "end": 11},
                "run":     {"start": 12, "end": 15}
            },
            # 为各组赋值
            "value": {
                "session":{
                    "001": 1,
                    "002": 2,
                },
                "task":{
                    "001": 1,
                    "002": 2,
                },
                "run":{
                    "001": 1,
                    "002": 2,
                },
            },
            # 最终会使用哪些组
            "use": ["task",],
        },
        # windowing 下必需配置的两项：use, method
        "windowing":{
            # 说明：这里仅定义“切窗策略的配置”，真正执行在 4_windowing.py
            "use": "single",  # 默认切窗方法，可选：events / single / sliding / events_single / events_sliding / single_sliding

            # 切窗方法。包括 cover 和 subdivide 两种。cover 重新切窗覆盖之前的数据，subdivide 选择一段数据切窗
            "method": "subdivide", # cover | subdivide

            # 切到哪些信号（不存在则自动跳过）
            "apply_to": ["rr","resp","acc"],

            # 统一默认值（单位一律为秒）
            "defaults": {
                "win_len_s": PARAMS["window_sec"],                 # 缺省窗长（用于需要窗长的模式）
                "bound_policy": "trim"                             # 超界时裁剪（不报错）
            },

            # 各模式参数（仅在被选中时读取）
            "modes": {
                # 1) 事件整段：按事件配对切段；不需要窗长
                "events": {
                    "events_path": "processed/norm/local",    # 事件文件路径；None 表示不支持
                    "pairs": [
                        ["baseline_start","stim_start"],
                        ["stim_start","stim_end"],
                        ["stim_end","intervention_start"],
                        ["intervention_start","intervention_end"]
                    ],
                },

                # 2) 单段：按绝对时间或锚点切一个段
                # 三选一：给 [start_s,end_s]；或 [start_s,win_len_s]；或 [anchor_time_s,win_len_s]
                # 时间按照绝对描述计算，可以从直接从上一层 level*/index.csv 
                # 里读 t_start_s 那一列的数值，按上面加减就行。
                # 比如 t_start_s = 739627.929，5分钟窗口 win_len = 300
                "single": {"start_s": None, "end_s": None, "win_len_s": 300, "anchor_time_s": 739627.929},

                # 3) 滑窗：在 [start_s,end_s] 范围内按 win_len/stride 切窗
                "sliding": {"start_s": None, "end_s": None, "win_len_s": None, "stride_s": None},

                # 4) 事件 + 单段：以事件为锚，局部切一个窗口
                "events_single": {
                    "events_path": "processed/norm/local",
                    "anchor_event": "stim_start",   # 选此模式，此项必填；None 则报错
                    "offset_s": 0.0,        # 相对锚点的偏移，秒；可为负
                    "win_len_s": None
                },

                # 5) 事件 + 滑窗：在某个事件区间内滑窗
                "events_sliding": {
                    "events_path": "processed/norm/local",
                    "segment": ["intervention_start","intervention_end"],  # 必填：事件名对
                    "win_len_s": None,
                    "stride_s":  None
                },

                # 6) 单段 + 滑窗：先定边界，再滑窗
                "single_sliding": {"start_s": None, "end_s": None, "win_len_s": None, "stride_s": None}
            },

            # 窗口含义命名模板（写入每个窗的 meaning 字段）
            "labeling": {
                "events":         "{e0}->{e1}",
                "single":         "single[{s:.1f},{e:.1f}]",
                "sliding":        "sliding[{s:.1f},{e:.1f}]/{w:.0f}",
                "events_single":  "{anchor}+{off:+.1f}s x {w:.0f}",
                "events_sliding": "{seg0}->{seg1} / {w:.0f}/{step:.0f}",
                "single_sliding": "range[{s:.1f},{e:.1f}] / {w:.0f}/{step:.0f}"
            },
        },
        # 输出参数
        "signal_features": [
            # -- 频域特征
            "hf_ms2", 
            "hf_log_ms2", 
            "lf_ms2", 
            "lf_log_ms2", 
            # "hf_band_used", 
            # "hf_center_hz",
            # -- 时域特征
            "mean_hr_bpm",
            "rmssd_ms",
            "sdnn_ms",
            "pnn50_pct",
            "sd1_ms",
            "sd2_ms",
            # rsa 特征
            "rsa_ms",
            "resp_rate_bpm",
            # "n_breaths_used",
            # "rsa_method",
            ],
        "preview_sids": [] # 选择预览被试id
    },
    "fantasia": {
        "loader": "scripts.loaders.fantasia_loader",
        "options": {
            "signals": ["ecg","resp"],       # 要落盘的信号
            "prefer_local_wfdb": True,       # 有 .hea/.dat/.ecg 就本地解析
            "allow_network": False,          # 禁止联网（你已有 subset）
            "cache_format": "parquet"        # 产 <sid>_ecg.parquet / <sid>_resp.parquet
        },
        "paths": {
            "raw":       "raw/fantasia/",
            "groups":   "groups/fantasia",  
            "norm":      "processed/norm/fantasia",
            "confirmed": "processed/confirmed/fantasia",
            "clean":        "processed/clean/fantasia",   # 检验并清理数据
            "preview":      "processed/preview/fantasia",
            "windowing": "processed/windowing/fantasia",  # 切窗
            "features":  "processed/features/fantasia",
            "final":    "processed/final/fantasia",        # 最终的数据表
        },
        "groups": {
            "sample": "f1o01 | f1y01",
            "group_map": {
                "age": {"start": 2, "end": 3},  # 左闭右开，0 起算
            },
            "value": {
                "age":{
                    "o": 1,
                    "y": 2,
                },
            },
            "use": ["age"],
        },
        "windowing":{
            "use": "single_sliding",  # 没有事件，默认滑窗
            "method": "cover", # cover | subdivide
            "apply_to": ["rr","resp"],
            "defaults": {
                "win_len_s": PARAMS["window_sec"],
                "bound_policy": "trim"
            },
            "modes": {
                "events": {"events_path": None, "path": None},
                "single": {"start_s": None, "end_s": None, "win_len_s": None, "anchor_time_s": None},
                "sliding": {"start_s": None, "end_s": None, "win_len_s": None, "stride_s": None},
                "events_single": {"events_path": None, "anchor_event": None, "offset_s": 0.0, "win_len_s": None},
                "events_sliding": {"events_path": None, "segment": None, "win_len_s": None, "stride_s": None},
                # 该数据没有 events，可以用 cover + single_sliding , 去掉开始的 2分钟，不设结尾，10分钟一个窗。
                "single_sliding": {"start_s": 120, "end_s": 999999, "win_len_s": 600, "stride_s": 0}
            },
            "labeling": {
                "events":         "{e0}->{e1}",
                "single":         "single[{s:.1f},{e:.1f}]",
                "sliding":        "sliding[{s:.1f},{e:.1f}]/{w:.0f}",
                "events_single":  "{anchor}+{off:+.1f}s x {w:.0f}",
                "events_sliding": "{seg0}->{seg1} / {w:.0f}/{step:.0f}",
                "single_sliding": "range[{s:.1f},{e:.1f}] / {w:.0f}/{step:.0f}"
            }
        },
        # 输出参数
        "signal_features": [
            # -- 频域特征
            "hf_ms2", 
            "hf_log_ms2", 
            "lf_ms2", 
            "lf_log_ms2", 
            # -- 时域特征
            "mean_hr_bpm",
            "rmssd_ms",
            "sdnn_ms",
            "pnn50_pct",
            "sd1_ms",
            "sd2_ms",
            # rsa 特征
            "rsa_ms",
            "resp_rate_bpm",
            ],
        "preview_sids": ["f2y05"] # 选择预览被试id
    }
    # "wesad": {...} 任何开源数据
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
# 信号别名（用于 detect_signal）。[] 可以加入任意可能的名字
SIGNAL_ALIASES = {
    "ecg":   ["ecg"],
    "rr":    ["rr","ibi","ppi"],
    "hr":    ["hr","bpm","heart_rate"],
    "resp":  ["resp","respiration","breath","breathing"],
    "ppg":   ["ppg","bvp"],
    "acc":   ["acc","accelerometer","accel"],
    "events":["markers", "events"]
}
DEVICE_TAGS = {"h10":"h", "verity":"v"}

# ---------- Column name schema (canonical output labels) ----------
SCHEMA = {
  # 每心跳
  "rr":     {"t": "t_s",      "v": "rr_ms"},
  "ppi":    {"t": "t_s",      "v": "rr_ms"},
  # 心率
  "hr":     {"t": "t_s",      "v": "bpm"},
  # 连续值
  "ecg":    {"t": "time_s",   "v":"value",    "fs":"fs_hz"},
  "resp":   {"t": "time_s",   "v": "value", "fs":"fs_hz"},
  "ppg":    {"t": "t_s",      "v": "value",   "fs":"fs_hz"},
  # 三轴加速度
  "acc":    {"t": "time_s",   "vx":"value_x", "vy":"value_y", "vz":"value_z"},
  # 事件标记
  "events": {"t": "time_s",   "label": "events"},
}

# ---------- Canonical event names (project-local vocabulary) ----------
EVENTS_CANON = [
  "baseline_start",
  "stim_start",
  "stim_end",
  "intervention_start",
  "intervention_end",
  "stop",
]

# ---------------------------------------------------------------

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