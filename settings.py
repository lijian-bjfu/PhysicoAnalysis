# 需要安装依赖：pip install wfdb pandas numpy scipy neurokit2 matplotlib

from pathlib import Path

# 指定数据集
ACTIVE_DATA = "local"

# 数据根
PROJECT_ROOT  = Path(__file__).parent
DATA_DIR      = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_CACHE_DIR = DATA_DIR / "raw" / ACTIVE_DATA

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
    # 下面的参数目的是让每个呼吸周期只出现一个稳定的主峰 
    # 0.5表示以该秒数为窗长对呼吸波做一次轻度平滑，再找峰
    "rsa_resp_smooth_sec": 0.5,
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
    
    # —— RR 短时尖峰清理参数（仅供 clean_rr 使用；不影响其他流程） ——
    # 最长“短时尖峰”连续长度（搏点数）。仅当 1 ≤ 段长 ≤ 本阈值时才自动修复，
    # 超出则只记录到 QC，不改变原始 RR。
    "rr_short_run_max_beats": 5,

    # 逐搏差的绝对阈值（毫秒）。|RR[i] − RR[i−1]| 大于该值将被视作候选异常，
    # 与相对偏差阈（rr_artifact_threshold）共同作用，增强鲁棒性。
    "rr_delta_abs_ms": 150.0,

    # “补偿判据”容差：|(RR_i + RR_{i+1}) − 2×局部中位 RR| / (2×局部中位 RR) < 本阈值
    # 视为“错分/多检”（一次心搏被拆成两搏），优先采用成对合并策略处理。
    "rr_pair_comp_tolerance": 0.15,

    # 提醒阈值：当检测到的“短时尖峰”段数 ≥ 本值时在日志与 QC 中给出提示，
    # 但不阻断也不改变修复策略（仅提醒数据质量一般）。
    "rr_groups_warn": 4,

    # 提醒阈值：需修正的搏点占比（相对于整条 RR 的长度）≥ 本值时提示，
    # 不中断。用于在批处理时快速筛出需关注的被试。
    "rr_max_correct_ratio_warn": 0.05,

    # 与参考 ecg_rr 比对时的“皆异常”判定阈值：在异常段的 ±2s 窗内，
    # 若 median(|RR_ecg − 中位| / 中位) 超过本值，则视为两路皆异常，仅入 QC 不做自动修复。
    "both_bad_rel_dev_thr": 0.15,

    # 插值方法（仅插值路径使用）。可选 "pchip"（单调分段三次，优先）或 "linear"。
    "interp_method": "pchip",

    # 是否启用“成对合并”修复（仅当命中补偿判据时生效）。关闭则所有短段一律走插值。
    "pair_merge_enable": True,

    # 设备 RR 与 ECG R 峰配对时的最近邻容差（毫秒）。仅用于 QC/可视化或后续扩展，
    # 不改变主要修复逻辑。
    "ecg_pair_tol_ms": 120.0,
}

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
            "rr_select": "processed/rr_select/local",  # 3_select_rr 的判断表位置
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
                "task": {"start": 3, "end": 4},
                # "session": {"start": 4, "end": 7},  # 左闭右开，0 起算
                # "task":    {"start": 8, "end": 11},
                # "run":     {"start": 12, "end": 15}
            },
            # 为各组赋值
            "value": {
                "task":{
                    "2": 1,
                    "3": 2,
                },
                # "session":{
                #     "001": 1,
                #     "002": 2,
                # },
                # "task":{
                #     "001": 1,
                #     "002": 2,
                # },
                # "run":{
                #     "001": 1,
                #     "002": 2,
                # },
            },
            # 最终会使用哪些组
            "use": ["task",],
        },
        # windowing 下必需配置的两项：use, method
        "windowing":{
            # 说明：这里仅定义“切窗策略的配置”，真正执行在 4_windowing.py
            "use": "events_offset",  # 默认切窗方法，可选：events / single / sliding / events_offset /events_single / events_sliding / single_sliding

            # 切窗方法。包括 cover 和 subdivide 两种。cover 重新切窗覆盖之前的数据，subdivide 选择一段数据切窗
            "method": "cover", # cover | subdivide

            # 切到哪些信号（不存在则自动跳过）
            "apply_to": ["rr","resp","acc"],

            # 各模式参数（仅在被选中时读取）
            "modes": {
                # 1) 事件整段：按事件配对切段；不需要窗长
                "events": {
                    "events_path": "processed/norm/local",    # 事件文件路径；None 表示不支持
                    "events_dict": {
                        1: "baseline_start",
                        2: "stim_start",
                        3: "stim_end",
                        4: "intervention_start",
                        5: "intervention_end",
                        6: "stop"
                    },
                },

                # 2) 单段：按绝对时间或锚点切一个段
                # 三选一：给 [start_s,end_s]；或 [start_s,win_len_s]；
                # 时间按照相对值设置，例如开始时间设为 12，
                # 会从数据开始位置向后偏移12秒作为切窗开始
                "single": {"start_s": None, "end_s": None, "win_len_s": 300},

                # 3) 滑窗：在 [start_s,end_s] 范围内按 win_len/stride 切窗
                # 时间按照相对值设置。例如开始时间设为 12, 
                # 则会从数据实际开始点向后偏移12秒作为起始点
                # win_len_s 不得为None
                # start_s / end_s 如果为 None 使用数据绝对起始点，同时向用户提出警告
                # stride_s 正数为两窗之间的间隔距离，复数表示两窗重叠距离。0或None表示紧贴
                "sliding": {"start_s": None, "end_s": None, "win_len_s": None, "stride_s": None},

                # 4) 事件+偏移：基于切窗数据对每个窗偏移避开不安全数据
                "events_offset": {
                    # 事件文件路径
                    "events_path": "processed/norm/local",
                    # 偏移设定
                    "offset": {
                        # 窗号:偏移值。窗号根据 events 定义，值由用户设定。
                        # 1:30 表示第一个窗向两头内缩进 30s
                        "1": 30,
                        "2": 60,
                    },
                },

                # 5) 事件 + 单段：以事件为锚，局部切一个窗口
                "events_single": {
                    "events_path": "processed/norm/local",
                    "anchor_event": "stim_start",   # 选此模式，此项必填；None 则报错
                    "offset_s": 0.0,        # 相对锚点的偏移，秒；可为负
                    "win_len_s": None
                },

                # 6) 事件 + 滑窗：在某个事件区间内滑窗
                "events_sliding": {
                    "events_path": "processed/norm/local",
                    "segment": ["intervention_start","intervention_end"],  # 必填：事件名对
                    "win_len_s": None,
                    "stride_s":  None
                },
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
        "preview_sids": ["P002S001T001R001"] # 选择预览被试id
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
  "custom_event",
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

# final.csv 输出列说明
FINAL_LABELS = {
    "subject_id": "被试唯一标识（字符串，例如 'sub-0001'）",
    "w_id": "窗口编号/序号",
    "w_s": "窗口开始时间，基于绝对值，第一个窗口开始时间为0",
    "w_e": "窗口开始时间，基于绝对值",
    "hf_log_ms2": "HF功率自然对数 ln(ms²)",
    "lf_log_ms2": "LF功率自然对数 ln(ms²)",
    "mean_hr_bpm": "平均心率（次/分）",
    "rmssd_ms": "RMSSD（毫秒）",
    "sdnn_ms": "SDNN（毫秒）",
    "pnn50_pct": "pNN50（百分比）",
    "sd1_ms": "SD1（毫秒）",
    "sd2_ms": "SD2（毫秒）",
    "rsa_ms": "RSA（毫秒；可缺）",
    "resp_rate_bpm": "呼吸频率（次/分；缺失则留空）",
    "rr_valid_ratio": "有效RR占比（0–1；≥0.80 推荐有效）",
    "rr_max_gap_s": "显示因时间轴缺口/掉峰造成的超长 RR",
    "group": "对照组",
}