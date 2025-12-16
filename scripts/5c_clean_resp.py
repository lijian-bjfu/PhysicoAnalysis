# scripts/4b_clean_resp.py
# 识别 resp 数据与窗口中rr数据是否长度一致，是否有缺失。保证与rr数据对齐
# 不完整的窗口 resp 数据会全部删除

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent))
        break
    _p = _p.parent
# --- end bootstrap ---

from settings import DATASETS, ACTIVE_DATA, DATA_DIR, SCHEMA, PARAMS  # type: ignore


# ==========================================================
# I/O（必须使用你现有的 settings 路径体系）
# ==========================================================
DS = DATASETS[ACTIVE_DATA]
paths = DS["paths"]
SRC_ROOT = (DATA_DIR / paths["windowing"]).resolve()   
# 呼吸数据命名格式为 SID_resp_窗口号.csv 或 .parquet，例如 "P001S001T001R001_resp_w01.csv"
# 注意：你的原代码里用到 SRC_ROOT 但未定义，这里按项目约定改为 DATA_DIR。
SRC_DIR = (SRC_ROOT / "collected" / "resp").resolve()
# 清理后的 resp 覆盖写回同目录
OUT_ROOT = SRC_DIR

# 切窗后的窗口数据报表位置（你给的文件名与目录结构）
windowing_report = (SRC_ROOT / "collected" / "collected_index.csv").resolve()

# 报表中的窗口号列（用于 resp 文件名 wXX）
w_id_col = "final_order"
# 报表中的窗口长度列（你给的表头是 duration_s；原代码拼写错误）
w_duration_col = "duration_s"


# ==========================================================
# SID 白名单：为空则处理全部被试
# ==========================================================
SID: List[str] = [
    "P001S001T001R001",
    "P002S001T002R001",
    "P003S001T001R001",
    "P004S001T002R001",
    "P006S001T002R001",
    "P007S001T001R001",
    "P008S001T002R001",
    "P009S001T001R001",
    "P010S001T002R001",
    "P011S001T001R001",
    "P012S001T001R001",
    "P013S001T002R001",
    "P014S001T001R001",
    "P015S001T002R001",
    "P016S001T001R001",
    "P017S001T001R001",
    "P018S001T001R001",
    "P019S001T001R001",
    "P020S001T001R001",
    "P021S001T001R001",
    "P022S001T001R001",
    "P023S001T001R001",
    "P024S001T002R001",
    "P025S001T002R001",
    "P026S001T002R001",
    "P027S001T002R001",
    "P028S001T002R001",
    "P029S001T002R001",
    "P030S001T002R001",
    "P031S001T002R001",
    "P032S001T001R001",
    "P033S001T001R001",
    "P034S001T001R001",
    "P035S001T002R001",
    "P036S001T002R001",
    "P037S001T002R001",
    "P038S001T001R001",
]
WIN = ['w01','w02','w03','w04','w05','w06']

# ==========================================================
# 清理策略开关（两种方法都实现，使用者可切换）
# ==========================================================
# "longest": 仅保留最长连续有效片段（推荐，最稳健，便于后续 RSA 单指标计算）
# "multi"  : 保留所有通过判定的连续有效片段（其余置 NaN；后续 RSA 计算需额外聚合策略）
CLEAN_MODE = str(PARAMS.get("resp_clean_mode", "multi"))


# ==========================================================
# 设计目标（按你最新约定的三段式流程）
# Step 1：最低工程门槛（不 reject，只切段）
#   - 时间戳严格递增
#   - 采样间隔在合理范围（围绕 50Hz，可抖动但不能有大段缺口）
#   - NaN/Inf 作为断点切段
# Step 2：生理结构（先判“像不像人呼吸”）
#   - 带通 0.10–0.50 Hz（6–30 次/分）
#   - 频域主峰 + 时域峰间期互证
#   - 最少有效时长 + 最少周期数
# Step 3：工程层面的“按周期删波”，而不是整窗判死
#   - 仅对已通过 Step 2 的候选片段进行
#   - 识别并剔除：夹紧/饱和点（0/1023）、长平台（近似直线）等
#   - 剔除后重新复核 Step 2
# ==========================================================

from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")  # 服务器/无GUI环境也能画图
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm


# ------------------------------
# 参数（尽量少，默认值尽量稳健）
# ------------------------------
FS_DEFAULT = float(PARAMS.get("resp_fs_hz", 50.0))
ADC_MIN = int(PARAMS.get("resp_adc_min", 0))
ADC_MAX = int(PARAMS.get("resp_adc_max", 1023))
ADC_EPS = int(PARAMS.get("resp_adc_eps", 0))  # 0 表示只认“精确贴边”

# Step 1（工程底线）
MAX_GAP_FACTOR = float(PARAMS.get("resp_max_gap_factor", 3.0))  # dt > factor * median_dt 视为缺口
MAX_MEDIAN_DT_DEV = float(PARAMS.get("resp_median_dt_dev", 0.35))  # |median_dt - 1/fs|/(1/fs) > dev 视为采样异常（仅记录，不直接 reject）

# Step 2（生理结构）
BAND_LO_HZ = float(PARAMS.get("resp_band_lo_hz", 0.10))
BAND_HI_HZ = float(PARAMS.get("resp_band_hi_hz", 0.50))
MIN_VALID_S = float(PARAMS.get("resp_min_valid_s", 12.0))
MIN_CYCLES = int(PARAMS.get("resp_min_cycles", 2))
PSD_NPERSEG_S = float(PARAMS.get("resp_psd_nperseg_s", 20.0))
PSD_PEAK_RATIO_MIN = float(PARAMS.get("resp_psd_peak_ratio_min", 0.20))  # 主峰能量占带内能量比例
# Step2：幅度/谱峰门槛（用于拦截“几乎常数/伪呼吸”）
P95P5_RAW_MIN = float(PARAMS.get("resp_ptp_raw_min", 20.0))   # 原始信号 P95-P5 最低幅度(ADC码值)
P95P5_BP_MIN  = float(PARAMS.get("resp_ptp_bp_min", 2.0))     # 带通后信号 P95-P5 最低幅度
PSD_PEAK_PROM_MIN = float(PARAMS.get("resp_psd_peak_prom_min", 1.0))  # 谱峰突出度阈值

# PRETRIM 的“塌陷/空白”识别不要复用 Step3 的 2s rolling 窗。
# 2s rolling 在“方波/削顶呼吸”的平顶段内会得到很小的 rolling ptp，从而把每一口呼吸的平顶误判为 collapse。
# PRETRIM 应该只切“长时间整体幅度塌陷”的段，因此 rolling 窗必须明显长于单次呼吸平顶（建议 >=8s）。
PRETRIM_ROLL_WIN_S = float(PARAMS.get("resp_pretrim_roll_win_s", 8.0))
# PRETRIM_COLLAPSE_ROLL_PTP_MAX: safer fallback, never reference undefined name
PRETRIM_COLLAPSE_ROLL_PTP_MAX = float(
    PARAMS.get(
        "resp_pretrim_collapse_roll_ptp_max",
        float(PARAMS.get("resp_collapse_roll_ptp_max", 10.0)),
    )
)

PERIOD_CV_MAX = float(PARAMS.get("resp_period_cv_max", 0.40))
PERIOD_CV_SOFT_MAX = float(PARAMS.get("resp_period_cv_soft_max", 0.35))


# Step 3（周期级工程剔除）
CYCLE_CLIP_FRAC_MAX = float(PARAMS.get("resp_cycle_clip_frac_max", 0.50))
CYCLE_FLAT_FRAC_MAX = float(PARAMS.get("resp_cycle_flat_frac_max", 0.50))
FLAT_EQ_EPS = float(PARAMS.get("resp_flat_eq_eps", 0.0))  # 0 表示严格相等；也可设为 1 允许 ±1 的“近似水平”
MIN_FLAT_RUN_S = float(PARAMS.get("resp_min_flat_run_s", 0.50))
MIN_CLIP_RUN_S = float(PARAMS.get("resp_min_clip_run_s", 0.20))
# “平直/幅度塌陷”用滚动峰峰值识别（解决 P001 初段近似常数但带轻微抖动、以及 P008 中段几乎无呼吸但有零星小峰的问题）
ROLL_WIN_S = float(PARAMS.get("resp_roll_win_s", 2.0))
FLAT_ROLL_PTP_MAX = float(PARAMS.get("resp_flat_roll_ptp_max", 2.0))
COLLAPSE_ROLL_PTP_MAX = float(PARAMS.get("resp_collapse_roll_ptp_max", 8.0))
MIN_COLLAPSE_RUN_S = float(PARAMS.get("resp_min_collapse_run_s", 2.0))

# Step2 前的“工程垃圾段”预剔除（只用于把长时间塌陷/锁死/贴边段切掉，避免整窗连坐）
# 注意：clip(0/1023) 在“削顶/方波式呼吸”里可能是正常现象（单次平顶可达 ~1s 甚至更久）。
# 因此：默认不把 clip 当作 Step2 前切段依据；如确需启用，请显式打开开关，并把阈值设得足够大。
PRETRIM_USE_CLIP = bool(PARAMS.get("resp_pretrim_use_clip", False))
# 重要：PRETRIM 的 clip 只用于切“真的锁死很久”的贴边段，必须显著大于单次正常平顶时长。
# 即便使用者把该值设得更小，函数内部也会强制下限为 3.0s。
PRETRIM_MIN_CLIP_RUN_S = float(PARAMS.get("resp_pretrim_min_clip_run_s", 3.0))
# 预剔除的核心：只切长时间“幅度塌陷/锁死”的段落
PRETRIM_MIN_COLLAPSE_RUN_S = float(PARAMS.get("resp_pretrim_min_collapse_run_s", MIN_COLLAPSE_RUN_S))

# Step 3（平台峰规范化：不删窗，只让平顶峰“中心点唯一最高”，便于下游稳定取相位）
PLATEAU_TOL = int(PARAMS.get("resp_plateau_tol", 1))
PLATEAU_MIN_LEN_S = float(PARAMS.get("resp_plateau_min_len_s", 0.08))
PLATEAU_SEARCH_S = float(PARAMS.get("resp_plateau_search_s", 0.60))
PLATEAU_DROP_DV = int(PARAMS.get("resp_plateau_drop_dv", 1))

# 输出
REPORT_DIR = (OUT_ROOT / "_resp_clean_reports").resolve()
PLOT_DIR = (OUT_ROOT / "_resp_clean_plots").resolve()
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------
# 数据结构
# ------------------------------
@dataclass
class RespCleanRow:
    subject_id: str
    final_order: int
    file: str
    status: str  # ok / partial / reject
    reasons: List[str]
    kept_spans: List[Tuple[float, float]]
    dropped_spans: List[Tuple[float, float]]

    expected_duration_s: Optional[float]

    # Step1 工程摘要
    time_median_dt: Optional[float]
    time_gap_frac: Optional[float]
    time_median_dt_dev: Optional[float]

    # Step2 生理摘要（基于最终保留片段）
    kept_frac: Optional[float]
    kept_dur_s: Optional[float]
    kept_cycles: Optional[int]
    br_bpm_median: Optional[float]
    period_cv: Optional[float]
    psd_peak_hz: Optional[float]
    psd_peak_ratio: Optional[float]
    psd_peak_prom: Optional[float]
    ptp_raw_p95p5: Optional[float]
    ptp_bp_p95p5: Optional[float]

    # Step3 工程证据（全窗统计，便于图上标注）
    clip_frac_all: Optional[float]
    clip_max_run_all: Optional[int]
    flat_max_run_all: Optional[int]
    collapse_frac_all: Optional[float]
    collapse_max_run_all: Optional[int]

    # Step2 复核信息（为可复核性：即使 reject，也记录每个尝试片段为什么失败）
    step2_checks: List[Dict[str, object]]
# ------------------------------
# Step2 前的预剔除 + subspan 切分
# ------------------------------

def _collapse_mask_roll(x: np.ndarray, fs: float, win_s: float, ptp_max: float) -> np.ndarray:
    """用于 PRETRIM 的“长时间幅度塌陷/空白”检测（点级 mask）。

    设计目标：
    - 只切“长时间整体幅度很小”的工程垃圾段（例如长期锁死在某个中间值、长期贴近常数、长期无呼吸）。
    - 不切“每一口呼吸的平顶/削顶”，因此 rolling 窗必须显著长于单次平顶。

    返回：collapse_mask，True 表示 rolling ptp 很小（疑似塌陷）。
    """
    n = len(x)
    if n < 3 or fs <= 0:
        return np.zeros_like(x, dtype=bool)

    win = int(max(3, round(win_s * fs)))
    s = pd.Series(x)
    rmax = s.rolling(win, center=True, min_periods=max(1, win // 4)).max().to_numpy()
    rmin = s.rolling(win, center=True, min_periods=max(1, win // 4)).min().to_numpy()
    rptp = rmax - rmin
    return np.isfinite(rptp) & (rptp <= float(ptp_max))

def _pretrim_bad_runs_and_subspans(t: np.ndarray, x: np.ndarray, fs: float) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Step2 前的预处理：只做“整窗连坐”的工程垃圾段预剔除，然后把剩余 True 的连续片段切成 subspans。

    设计原则（为稳定性服务，而不是为了“看起来很聪明”）：
    - 预剔除只负责移除“长时间不可用”的段落（例如长时间幅度塌陷/锁死，或极长时间贴边）。
    - 预剔除**不**把每一口呼吸的正常平顶/削顶当成垃圾段，否则会把方波式呼吸切碎成大量短片段，
      进而在 Step2 被判定为 too_short，导致看起来“算法很不稳定”。

    返回：
      - subspans: (start, end) 相对索引，左闭右开
      - keep_local: 长度与 x 相同的 bool mask

    目标：像 P001 这种“前面长时间塌陷，尾部有可用呼吸”的窗口，不要被整窗连坐。
    """
    n = len(x)
    keep = np.isfinite(x) & np.isfinite(t)
    if n < 3 or not np.any(keep):
        return [], keep

    # ---- 1) 预剔除：只处理“长时间塌陷/锁死/贴边很久”的段落 ----
    # PRETRIM 只做“长时间整体塌陷/空白”的切段。
    # 不要复用 Step3 的 flat/collapse（2s rolling）逻辑，否则会把方波/削顶呼吸的平顶误切。
    collapse_mask = _collapse_mask_roll(
        x=x,
        fs=fs,
        win_s=PRETRIM_ROLL_WIN_S,
        ptp_max=PRETRIM_COLLAPSE_ROLL_PTP_MAX,
    )

    # 可选：只切“非常长”的贴边段。默认关闭；即便开启，也必须让阈值显著大于单次正常平顶时长。
    if PRETRIM_USE_CLIP:
        min_clip_run = int(round(max(PRETRIM_MIN_CLIP_RUN_S, 3.0) * fs))
        clip = _clip_mask(x)
        for a, b in _runs_bool(clip):
            if (b - a) >= min_clip_run:
                keep[a:b] = False

    # 只切“长时间幅度塌陷”的段落（默认 >=2s）
    min_collapse_run = int(round(PRETRIM_MIN_COLLAPSE_RUN_S * fs))
    for a, b in _runs_bool(collapse_mask):
        if (b - a) >= min_collapse_run:
            keep[a:b] = False

    # ---- 2) 生成 subspans：keep==True 的连续段 ----
    subspans: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        while i < n and not keep[i]:
            i += 1
        if i >= n:
            break
        j = i + 1
        while j < n and keep[j]:
            j += 1
        # 只做最小长度保护，避免极短碎片
        if j - i >= 5:
            subspans.append((i, j))
        i = j

    return subspans, keep


# ------------------------------
# I/O 帮助函数
# ------------------------------

def _guess_time_value_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    # 常见命名
    time_cand = [c for c in cols if c.lower() in {"t", "time", "time_s", "timestamp", "ts", "unix", "t_s"}]
    val_cand = [c for c in cols if c.lower() in {"resp", "value", "x", "y"}]

    if time_cand and val_cand:
        return time_cand[0], val_cand[0]

    # 退化：找前两个数值列
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        return num_cols[0], num_cols[1]

    raise ValueError(f"Cannot infer time/value columns from columns={cols}")


def _read_resp_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported resp file: {path}")


def _write_resp_file(df: pd.DataFrame, path: Path) -> None:
    # 覆盖写回同路径：保持原格式
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported resp file: {path}")


def _load_windowing_report() -> pd.DataFrame:
    rep = pd.read_csv(windowing_report)
    # 兼容你之前的字段
    need = {"subject_id", w_id_col}
    for k in need:
        if k not in rep.columns:
            raise ValueError(f"collected_index.csv missing column: {k}")
    return rep


def _list_window_files() -> List[Path]:
    files: List[Path] = []
    for p in sorted(SRC_DIR.glob("*resp_w*")):
        if p.suffix.lower() not in {".csv", ".parquet", ".pq"}:
            continue
        files.append(p)
    return files


def _parse_sid_wid_from_name(name: str) -> Tuple[str, str]:
    # 例如 P001S001T001R001_resp_w03.csv
    stem = name
    for suf in [".csv", ".parquet", ".pq"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    if "_resp_" not in stem:
        raise ValueError(f"Bad resp window filename: {name}")
    sid, wpart = stem.split("_resp_", 1)
    # wpart like w01
    return sid, wpart


# ------------------------------
# Step 1：最低工程门槛（切段，不 reject）
# ------------------------------

def _split_by_time_and_finite(t: np.ndarray, x: np.ndarray, fs: float) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    """返回候选片段（索引区间，左闭右开）以及工程摘要指标。"""
    n = len(t)
    if n < 3:
        return [], {"median_dt": np.nan, "gap_frac": np.nan, "median_dt_dev": np.nan}

    # 时间递增检查：不递增点作为断点
    dt = np.diff(t)
    non_increasing = dt <= 0

    median_dt = float(np.nanmedian(dt))
    expected_dt = 1.0 / fs
    median_dt_dev = float(abs(median_dt - expected_dt) / expected_dt) if np.isfinite(median_dt) else np.nan

    # 缺口检查：dt 远大于中位数视为断点
    gap = dt > (MAX_GAP_FACTOR * median_dt) if np.isfinite(median_dt) and median_dt > 0 else np.ones_like(dt, dtype=bool)
    gap_frac = float(np.mean(gap)) if len(gap) else 0.0

    # 有效值检查
    finite = np.isfinite(x) & np.isfinite(t)

    # 断点：任何一项为 True，都切
    cut = np.zeros(n, dtype=bool)
    # dt 基于边 i->i+1，所以把断点放在 i+1
    cut[1:][non_increasing] = True
    cut[1:][gap] = True
    # 非有限点自身作为断点
    cut[~finite] = True

    # 将 cut 转为片段
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        # 跳过断点本身
        while start < n and cut[start]:
            start += 1
        if start >= n:
            break
        end = start + 1
        while end < n and not cut[end]:
            end += 1
        spans.append((start, end))
        start = end

    info = {"median_dt": median_dt, "gap_frac": gap_frac, "median_dt_dev": median_dt_dev}
    return spans, info


# ------------------------------
# Step 2：生理结构（先判“像不像呼吸”）
# ------------------------------

def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999999)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x)


def _psd_peak(x: np.ndarray, fs: float) -> Tuple[float, float, float]:
    """返回 (peak_hz, peak_ratio_in_band, peak_prom).

    - peak_ratio: 主峰功率 / 频带(0.10–0.50Hz)总能量
    - peak_prom : 主峰相对频带中位功率的突出度 (peak_power - median) / median
    """
    nperseg = int(max(8, min(len(x), PSD_NPERSEG_S * fs)))
    f, pxx = welch(x, fs=fs, nperseg=nperseg)
    band = (f >= BAND_LO_HZ) & (f <= BAND_HI_HZ)
    if not np.any(band):
        return np.nan, np.nan, np.nan
    fb = f[band]
    pb = pxx[band]
    if len(pb) < 3 or not np.all(np.isfinite(pb)):
        return np.nan, np.nan, np.nan

    i = int(np.argmax(pb))
    peak_hz = float(fb[i])
    peak_power = float(pb[i])

    band_power = float(np.trapz(pb, fb))
    band_width = float(fb[-1] - fb[0]) if len(fb) >= 2 else np.nan
    band_mean = float(band_power / band_width) if np.isfinite(band_power) and np.isfinite(band_width) and band_width > 0 else np.nan
    ratio = float(peak_power / band_mean) if np.isfinite(band_mean) and band_mean > 0 else np.nan

    med_pb = float(np.median(pb))
    prom = float((peak_power - med_pb) / med_pb) if med_pb > 0 else np.nan

    return peak_hz, ratio, prom


def _breath_peaks(
    x_f: np.ndarray,
    fs: float,
    expected_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """在带通后信号上找峰与谷，用于周期估计。

    关键改动：用 PSD 主峰频率推导“期望周期”，自适应设置最小峰距，避免削顶/方波式波形在同一口气里被重复计峰。

    - 期望周期 T = 1/expected_hz
    - 最小峰距 min_dist_s = clamp(0.5*T, 1s, 4s)
    - 若 expected_hz 不可用，则退化为 1s
    """
    # 默认：至少 1s（避免噪声造峰）
    min_dist_s = 1.0

    if expected_hz is not None and np.isfinite(expected_hz) and expected_hz > 0:
        T = 1.0 / float(expected_hz)
        min_dist_s = 0.5 * T
        # clamp 到 [1s, 4s]，避免极端值
        min_dist_s = float(np.clip(min_dist_s, 1.0, 4.0))

    min_dist = int(max(1, round(fs * min_dist_s)))
    pks, _ = find_peaks(x_f, distance=min_dist)
    vls, _ = find_peaks(-x_f, distance=min_dist)
    return pks, vls


def _period_stats(t: np.ndarray, peak_idx: np.ndarray) -> Tuple[float, float, int]:
    if len(peak_idx) < 2:
        return np.nan, np.nan, 0

    peak_t = t[peak_idx]
    periods = np.diff(peak_t)
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if len(periods) < 1:
        return np.nan, np.nan, 0

    # --- Robustification ---
    # Square/plateau-like respiration and occasional breath-holds can create a few very long/short intervals.
    # Use a trimmed + MAD-based CV so Step2 is stable and does not overreact to a small number of outliers.
    per = periods.copy()
    if len(per) >= 10:
        lo = np.quantile(per, 0.10)
        hi = np.quantile(per, 0.90)
        per = per[(per >= lo) & (per <= hi)]
        if len(per) < 3:
            per = periods  # fallback

    med = float(np.median(per))
    if not np.isfinite(med) or med <= 0:
        return np.nan, np.nan, 0

    # Classic CV (still useful)
    cv = float(np.std(per) / med)

    # Robust CV using MAD (1.4826*MAD approximates std for normal data)
    mad = float(np.median(np.abs(per - med)))
    rcv = float((1.4826 * mad) / med) if mad >= 0 else np.nan

    # Use the more conservative (larger) of the two if both are finite;
    # this avoids accidentally passing purely noisy signals.
    if np.isfinite(cv) and np.isfinite(rcv):
        cv_use = float(max(cv, rcv))
    else:
        cv_use = float(cv) if np.isfinite(cv) else float(rcv)

    return med, cv_use, int(len(per) - 1)


def _is_physio_valid(t: np.ndarray, x_raw: np.ndarray, fs: float) -> Tuple[bool, Dict[str, float], np.ndarray]:
    """返回 (是否有效, 指标, 带通后的信号).

    说明：这里属于 Step2（生理结构门槛）。在进入谱峰/峰间期之前，先做最便宜的“幅度门槛”，
    用来拦截 P017/P018 这类“几乎常数/伪呼吸”信号（即便 find_peaks 也能硬造周期）。
    """
    dur = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    if dur < MIN_VALID_S:
        return False, {"dur": dur, "tag": "too_short"}, np.full_like(x_raw, np.nan)

    # Step2-A：原始幅度门槛（P95-P5）
    p95 = float(np.nanpercentile(x_raw, 95))
    p05 = float(np.nanpercentile(x_raw, 5))
    ptp_raw = float(p95 - p05)
    if not np.isfinite(ptp_raw) or ptp_raw < P95P5_RAW_MIN:
        return False, {"dur": dur, "ptp_raw_p95p5": ptp_raw, "tag": "too_small_amplitude_raw"}, np.full_like(x_raw, np.nan)

    # 去中心化再滤波（降低直流漂移影响）
    x0 = x_raw - np.nanmedian(x_raw)
    try:
        xf = _bandpass(x0, fs=fs, lo=BAND_LO_HZ, hi=BAND_HI_HZ)
    except Exception:
        return False, {"dur": dur, "ptp_raw_p95p5": ptp_raw, "tag": "bandpass_failed"}, np.full_like(x_raw, np.nan)

    # Step2-B：带通后幅度门槛（避免“几乎常数 + 数值抖动”伪装成周期）
    p95b = float(np.nanpercentile(xf, 95))
    p05b = float(np.nanpercentile(xf, 5))
    ptp_bp = float(p95b - p05b)
    if not np.isfinite(ptp_bp) or ptp_bp < P95P5_BP_MIN:
        return False, {"dur": dur, "ptp_raw_p95p5": ptp_raw, "ptp_bp_p95p5": ptp_bp, "tag": "too_small_amplitude_bp"}, xf

    peak_hz, peak_ratio, peak_prom = _psd_peak(xf, fs=fs)
    if not (np.isfinite(peak_hz) and np.isfinite(peak_ratio) and np.isfinite(peak_prom)):
        return False, {
            "dur": dur,
            "ptp_raw_p95p5": ptp_raw,
            "ptp_bp_p95p5": ptp_bp,
            "psd_peak_hz": peak_hz,
            "psd_peak_ratio": peak_ratio,
            "psd_peak_prom": peak_prom,
            "tag": "psd_invalid",
        }, xf

    # 时域：用“峰”来估计周期（峰更稳定，谷也可以备用）
    pks, vls = _breath_peaks(xf, fs=fs, expected_hz=peak_hz)
    per_med, per_cv, n_cycles = _period_stats(t, pks)
    # Note: per_cv here is already a robustified CV (trimmed + MAD-backed), see _period_stats.
    if n_cycles < MIN_CYCLES:
        # 峰太少，用谷再试一次
        per_med2, per_cv2, n_cycles2 = _period_stats(t, vls)
        if n_cycles2 > n_cycles:
            per_med, per_cv, n_cycles = per_med2, per_cv2, n_cycles2

    if not np.isfinite(per_med) or n_cycles < MIN_CYCLES:
        return False, {
            "dur": dur,
            "ptp_raw_p95p5": ptp_raw,
            "ptp_bp_p95p5": ptp_bp,
            "psd_peak_hz": peak_hz,
            "psd_peak_ratio": peak_ratio,
            "psd_peak_prom": peak_prom,
            "n_cycles": n_cycles,
            "tag": "too_few_cycles",
        }, xf

    br_bpm = 60.0 / per_med if per_med > 0 else np.nan

    # 两条互证：频域主峰对应的 bpm 与时域 bpm 不能差太离谱
    br_bpm_psd = 60.0 * peak_hz if np.isfinite(peak_hz) else np.nan
    agree = (np.isfinite(br_bpm) and np.isfinite(br_bpm_psd) and abs(br_bpm - br_bpm_psd) <= 6.0)

    # 逐条判断，给出更可复核的失败原因（避免所有失败都归为 physio_failed）
    fail_tag = "ok"
    if not (BAND_LO_HZ <= peak_hz <= BAND_HI_HZ):
        fail_tag = "psd_peak_out_of_band"
    elif peak_ratio < PSD_PEAK_RATIO_MIN:
        fail_tag = "psd_peak_ratio_low"
    elif peak_prom < PSD_PEAK_PROM_MIN:
        # 谱峰不够突出通常表示频带内能量较分散（噪声/运动伪迹/非平稳）。
        # 但在“时域节律非常稳定”的情况下（周期足够多、CV较低、且频域/时域bpm一致），
        # 允许作为弱证据通过，避免把明显有呼吸节律的长窗整窗判死。
        strong_time_rhythm = (
            (n_cycles >= max(MIN_CYCLES, 6))
            and np.isfinite(per_cv)
            and (per_cv <= min(PERIOD_CV_MAX, 0.25))
            and bool(agree)
        )
        if strong_time_rhythm:
            fail_tag = "ok_weak_spectral_peak"
        else:
            fail_tag = "psd_peak_prom_low"
    elif not (BAND_LO_HZ * 60.0 <= br_bpm <= BAND_HI_HZ * 60.0):
        fail_tag = "bpm_out_of_band"
    elif per_cv > PERIOD_CV_MAX:
        # Human respiration can be irregular (pause/hold) while still having a strong respiratory rhythm.
        # If frequency-domain evidence is strong and the PSD/time-domain rates agree, allow a soft pass
        # up to a slightly higher CV threshold; otherwise keep the strict reject.
        strong_freq_rhythm = (
            (n_cycles >= max(MIN_CYCLES, 6))
            and bool(agree)
            and np.isfinite(peak_ratio)
            and np.isfinite(peak_prom)
            and (peak_ratio >= PSD_PEAK_RATIO_MIN)
            and (peak_prom >= PSD_PEAK_PROM_MIN)
        )
        if strong_freq_rhythm and np.isfinite(per_cv) and (per_cv <= max(PERIOD_CV_SOFT_MAX, PERIOD_CV_MAX)):
            fail_tag = "ok_noisy_period"
        else:
            fail_tag = "period_cv_high"
    elif not agree:
        fail_tag = "bpm_psd_disagree"

    ok = (fail_tag == "ok" or fail_tag == "ok_weak_spectral_peak" or fail_tag == "ok_noisy_period")

    metrics = {
        "dur": dur,
        "ptp_raw_p95p5": float(ptp_raw),
        "ptp_bp_p95p5": float(ptp_bp),
        "psd_peak_hz": float(peak_hz),
        "psd_peak_ratio": float(peak_ratio),
        "psd_peak_prom": float(peak_prom),
        "br_bpm": float(br_bpm),
        "br_bpm_psd": float(br_bpm_psd),
        "period_cv": float(per_cv),
        "period_cv_max": float(PERIOD_CV_MAX),
        "period_cv_soft_max": float(PERIOD_CV_SOFT_MAX),
        "n_cycles": float(n_cycles),
        "tag": (fail_tag if ok else fail_tag),
    }
    # 保留 ok_weak_spectral_peak 作为“通过但谱峰较弱”的可复核标签
    if ok and fail_tag == "ok":
        metrics["tag"] = "ok"
    return bool(ok), metrics, xf


# ------------------------------
# Step 3：平台峰规范化（只在 Step2 通过的片段内做；不再“切碎/删周期”）
# ------------------------------


# Plateau peak normalization helpers
def _canonicalize_plateau_peaks_segment(t: np.ndarray, x_seg: np.ndarray, xf_seg: np.ndarray, fs: float) -> Tuple[np.ndarray, List[Tuple[float, float]], List[float]]:
    x_out = x_seg.copy()
    plateau_spans: List[Tuple[float, float]] = []
    plateau_centers: List[float] = []
    if x_seg.size < 5 or not np.any(np.isfinite(x_seg)):
        return x_out, plateau_spans, plateau_centers

    # 用该片段自身的 PSD 主峰推导最小峰距，减少削顶/方波导致的重复计峰
    peak_hz_local, _, _ = _psd_peak(xf_seg, fs=fs)
    pks, _ = _breath_peaks(xf_seg, fs=fs, expected_hz=peak_hz_local)
    if pks is None or len(pks) == 0:
        return x_out, plateau_spans, plateau_centers

    win = int(max(2, round(PLATEAU_SEARCH_S * fs)))
    min_len = int(max(2, round(PLATEAU_MIN_LEN_S * fs)))

    for p in pks:
        p = int(p)
        a0 = max(0, p - win)
        b0 = min(len(x_out), p + win + 1)
        xs = x_out[a0:b0]
        if xs.size < 3 or not np.any(np.isfinite(xs)):
            continue

        vmax = float(np.nanmax(xs))
        if not np.isfinite(vmax):
            continue
        near = np.isfinite(xs) & (np.abs(xs - vmax) <= float(PLATEAU_TOL))
        if not np.any(near):
            continue

        runs = _runs_bool(near)
        if not runs:
            continue
        ra, rb = max(runs, key=lambda ab: ab[1] - ab[0])
        if (rb - ra) < min_len:
            continue

        g_a = a0 + ra
        g_b = a0 + rb
        c = (g_a + g_b - 1) // 2

        plateau_spans.append((float(t[g_a]), float(t[g_b - 1])))
        plateau_centers.append(float(t[c]))

        for i in range(g_a, g_b):
            if i == c:
                continue
            if not np.isfinite(x_out[i]):
                continue
            x_out[i] = max(float(ADC_MIN), float(x_out[i] - PLATEAU_DROP_DV))

    return x_out, plateau_spans, plateau_centers


def _scale_bandpass_to_raw(t: np.ndarray, x_raw: np.ndarray, xf: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y = np.full_like(x_raw, np.nan, dtype=float)
    m = mask & np.isfinite(x_raw) & np.isfinite(xf)
    if not np.any(m):
        return y

    xr = x_raw[m]
    xb = xf[m]
    xr_amp = float(np.nanpercentile(xr, 95) - np.nanpercentile(xr, 5))
    xb_amp = float(np.nanpercentile(xb, 95) - np.nanpercentile(xb, 5))
    if not np.isfinite(xr_amp) or xr_amp <= 0 or not np.isfinite(xb_amp) or xb_amp <= 0:
        return y

    xr_med = float(np.nanmedian(xr))
    scale = 0.5 * xr_amp / xb_amp
    y[m] = xr_med + (xf[m] * scale)
    return y

def _clip_mask(x: np.ndarray) -> np.ndarray:
    lo = ADC_MIN + ADC_EPS
    hi = ADC_MAX - ADC_EPS
    return (x <= lo) | (x >= hi)


def _max_run_bool(mask: np.ndarray) -> int:
    """返回 True 的最长连续长度。"""
    if mask.size == 0:
        return 0
    m = mask.astype(np.int8)
    # run-length: 找到 True 段起止
    diff = np.diff(np.r_[0, m, 0])
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))

def _runs_bool(mask: np.ndarray) -> List[Tuple[int, int]]:
    """返回 True 的所有连续 runs（左闭右开）。"""
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    diff = np.diff(np.r_[0, m, 0])
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]


def _flat_runs(x: np.ndarray, eps: float, fs: float) -> Tuple[np.ndarray, int, np.ndarray]:
    """返回 (flat_mask, max_run_len, collapse_mask).

    - flat_mask: “几乎不变/近似直线”的点（用于删除平直段）
    - collapse_mask: “幅度塌陷”的点（滚动峰峰值很小，允许有零星小峰/抖动）

    说明：
    - 仅用 `|dx|<=eps` 会漏掉“轻微抖动但总体平直”的段（P001 的初段）。
    - 仅用 ADC 贴边也会漏掉“在中间值附近锁死/塌陷”的段（P017/P018/P008 的中段）。
    因此这里引入滚动窗口峰峰值(rolling peak-to-peak)作为更稳健的“近似常数/塌陷”证据。
    """
    n = len(x)
    if n < 2:
        m = np.zeros_like(x, dtype=bool)
        return m, 0, m

    # 1) 差分近似相等（点级）
    dx = np.diff(x)
    eq_edge = np.abs(dx) <= eps
    eq_mask = np.zeros(n, dtype=bool)
    eq_mask[:-1] |= eq_edge
    eq_mask[1:] |= eq_edge

    # 2) 滚动峰峰值（点级）
    win = int(max(3, round(ROLL_WIN_S * fs)))
    # 用 pandas rolling 简化实现（本脚本已引入 pandas）
    s = pd.Series(x)
    rmax = s.rolling(win, center=True, min_periods=max(1, win // 4)).max().to_numpy()
    rmin = s.rolling(win, center=True, min_periods=max(1, win // 4)).min().to_numpy()
    rptp = rmax - rmin

    collapse_mask = np.isfinite(rptp) & (rptp <= COLLAPSE_ROLL_PTP_MAX)
    flat_mask = eq_mask | (np.isfinite(rptp) & (rptp <= FLAT_ROLL_PTP_MAX))

    return flat_mask, _max_run_bool(flat_mask), collapse_mask


def _cycle_bounds_from_peaks(t: np.ndarray, xf: np.ndarray, fs: float) -> List[Tuple[int, int]]:
    pks, _ = _breath_peaks(xf, fs=fs)
    if len(pks) < 2:
        return []
    cycles: List[Tuple[int, int]] = []
    for i in range(len(pks) - 1):
        a, b = int(pks[i]), int(pks[i + 1])
        if b > a + 2:
            cycles.append((a, b))
    return cycles


def _mask_bad_cycles(t: np.ndarray, x: np.ndarray, xf: np.ndarray, fs: float) -> Tuple[np.ndarray, List[Tuple[float, float]], Dict[str, float]]:
    """返回 (keep_mask, removed_spans_time, quality_summary).

    规则：
    - 先做“与周期无关”的删除：长时间贴边(clip)、长时间平直(flat)、长时间幅度塌陷(collapse)。
      这一步解决：
      - P001 初段几乎常数但带细小抖动（不是严格相等，旧 flat 检测会漏掉）
      - P008 中段几乎无呼吸，仅有零星小峰（需要按幅度塌陷切掉）
      - P017/P018 类“中间值附近锁死/二值跳变”（幅度很小，应该整体不算呼吸）
    - 再做“周期级删除”：对剩余信号按周期检查 clip/flat 比例。
    """
    n = len(x)
    keep = np.ones(n, dtype=bool)
    removed: List[Tuple[float, float]] = []

    clip = _clip_mask(x)
    flat_mask, flat_max_run, collapse_mask = _flat_runs(x, eps=FLAT_EQ_EPS, fs=fs)

    min_flat_run = int(round(MIN_FLAT_RUN_S * fs))
    min_clip_run = int(round(MIN_CLIP_RUN_S * fs))
    min_collapse_run = int(round(MIN_COLLAPSE_RUN_S * fs))

    # --- 3-A) 与周期无关的“长 run”删除 ---
    for a, b in _runs_bool(clip):
        if (b - a) >= min_clip_run:
            keep[a:b] = False
            removed.append((float(t[a]), float(t[b - 1])))

    for a, b in _runs_bool(flat_mask):
        if (b - a) >= min_flat_run:
            keep[a:b] = False
            removed.append((float(t[a]), float(t[b - 1])))

    for a, b in _runs_bool(collapse_mask):
        if (b - a) >= min_collapse_run:
            keep[a:b] = False
            removed.append((float(t[a]), float(t[b - 1])))

    # --- 3-B) 周期级删除（只对仍保留的部分做） ---
    cycles = _cycle_bounds_from_peaks(t, xf, fs)
    for (a, b) in cycles:
        seg = slice(a, b)
        # 若该周期大部分已被上一步删除，就不再重复处理
        if np.mean(keep[seg]) < 0.5:
            continue

        nseg = max(1, b - a)
        clip_frac = float(np.mean(clip[seg]))
        flat_frac = float(np.mean(flat_mask[seg]))
        flat_run = _max_run_bool(flat_mask[seg])

        bad = False
        if clip_frac >= CYCLE_CLIP_FRAC_MAX:
            bad = True
        if flat_frac >= CYCLE_FLAT_FRAC_MAX and flat_run >= min_flat_run:
            bad = True

        if bad:
            keep[seg] = False
            removed.append((float(t[a]), float(t[b - 1])))

    summary = {
        "clip_frac_all": float(np.mean(clip)) if n else np.nan,
        "clip_max_run_all": int(_max_run_bool(clip)),
        "flat_max_run_all": int(flat_max_run),
        "collapse_frac_all": float(np.mean(collapse_mask)) if n else np.nan,
        "collapse_max_run_all": int(_max_run_bool(collapse_mask)),
    }
    return keep, removed, summary


def _mask_to_spans(t: np.ndarray, keep: np.ndarray) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """把 keep mask 转为 kept_spans/dropped_spans（时间区间）。"""
    n = len(keep)
    if n == 0:
        return [], []

    def runs(val: bool) -> List[Tuple[int, int]]:
        m = keep if val else ~keep
        diff = np.diff(np.r_[0, m.astype(np.int8), 0])
        s = np.where(diff == 1)[0]
        e = np.where(diff == -1)[0]
        return [(int(a), int(b)) for a, b in zip(s, e) if b > a]

    kept_idx = runs(True)
    drop_idx = runs(False)

    kept_spans = [(float(t[a]), float(t[b - 1])) for a, b in kept_idx]
    dropped_spans = [(float(t[a]), float(t[b - 1])) for a, b in drop_idx]
    return kept_spans, dropped_spans


# ------------------------------
# 可视化证据（你要求“所有判无效的证据都画出来”）
# ------------------------------

def _plot_debug(
    out_png: Path,
    sid: str,
    wtag: str,
    t: np.ndarray,
    x_raw: np.ndarray,
    x_clean: np.ndarray,
    keep_mask: np.ndarray,
    xf_keep: Optional[np.ndarray],
    spans_step1: List[Tuple[int, int]],
    plateau_spans: List[Tuple[float, float]],
    plateau_centers: List[float],
    band_mask: Optional[np.ndarray] = None,
) -> None:
    """两行图：
    - 上：原始数据（红框=最终被删除区间；橙框=平顶被修改区间；灰线=带通后“像呼吸”的部分叠加）
    - 下：清理后的数据（最终写回结果）
    """
    fig = plt.figure(figsize=(14, 7))

    # ---------- 上：raw + 证据标注 ----------
    ax = fig.add_subplot(2, 1, 1)
    ax.set_title(f"RESP clean debug | {sid} {wtag}")
    ax.plot(t, x_raw, linewidth=1.0, label="raw", zorder=2)

    # Step1 切段边界（淡虚线）
    for (a, b) in spans_step1:
        if 0 <= a < len(t):
            ax.axvline(t[a], linestyle="--", alpha=0.15)
        if 0 < b <= len(t):
            ax.axvline(t[b - 1], linestyle="--", alpha=0.15)

    # 叠加“像呼吸”的带通波形（仅用于可视化，缩放到 raw 尺度）
    if xf_keep is not None and np.any(np.isfinite(xf_keep)):
        m = band_mask if band_mask is not None else keep_mask
        xf_scaled = _scale_bandpass_to_raw(t, x_raw, xf_keep, m)
        if np.any(np.isfinite(xf_scaled)):
            ax.plot(
                t, xf_scaled,
                linewidth=1.0, alpha=0.35, color="gray",
                label="bandpass(human)", zorder=1
            )

    # 删除区间（红框）：由 keep_mask 的反区间得到
    _, dropped_sp = _mask_to_spans(t, keep_mask)
    ymin, ymax = ax.get_ylim()
    for (ts, te) in dropped_sp:
        if not (np.isfinite(ts) and np.isfinite(te) and te > ts):
            continue
        rect = Rectangle(
            (ts, ymin), te - ts, ymax - ymin,
            fill=False, edgecolor="red", linewidth=1.6, alpha=0.85, zorder=5
        )
        ax.add_patch(rect)

    # 平顶被修改（橙框）与中心点
    for (ts, te) in plateau_spans:
        if not (np.isfinite(ts) and np.isfinite(te) and te > ts):
            continue
        rect = Rectangle(
            (ts, ymin), te - ts, ymax - ymin,
            fill=False, edgecolor="orange", linewidth=1.2, alpha=0.85, zorder=6
        )
        ax.add_patch(rect)

    if plateau_centers:
        ct = np.array([c for c in plateau_centers if np.isfinite(c)], dtype=float)
        if ct.size:
            idx = np.clip(np.searchsorted(t, ct), 0, len(t) - 1)
            ax.scatter(t[idx], x_raw[idx], s=18, alpha=0.85, label="plateau_center", zorder=7)

    ax.set_ylabel("RESP (raw)")
    ax.legend(loc="upper right")

    # ---------- 下：cleaned ----------
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax)
    ax2.plot(t, x_clean, linewidth=1.0, label="cleaned", zorder=2)

    if plateau_centers:
        ct = np.array([c for c in plateau_centers if np.isfinite(c)], dtype=float)
        if ct.size:
            idx = np.clip(np.searchsorted(t, ct), 0, len(t) - 1)
            ax2.scatter(t[idx], x_clean[idx], s=18, alpha=0.85, label="plateau_center", zorder=3)

    ax2.set_ylabel("RESP (cleaned)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ------------------------------
# 单窗处理
# ------------------------------

def _process_one_window(
    path: Path,
    expected_dur: Optional[float],
    sid: str,
    wtag: str,
) -> RespCleanRow:
    df = _read_resp_file(path)
    tcol, vcol = _guess_time_value_cols(df)
    t = df[tcol].to_numpy(dtype=float)
    x = df[vcol].to_numpy(dtype=float)

    fs = FS_DEFAULT

    reasons: List[str] = []
    step2_checks: List[Dict[str, object]] = []
    tested_mask = np.zeros_like(x, dtype=bool)
    xf_tested = np.full_like(x, np.nan, dtype=float)

    # Step 1：切段
    spans, info1 = _split_by_time_and_finite(t, x, fs)
    median_dt = info1["median_dt"]
    gap_frac = info1["gap_frac"]
    median_dt_dev = info1["median_dt_dev"]

    if not np.isfinite(median_dt) or len(spans) == 0:
        reasons.append("empty_or_invalid_time")
        keep_mask = np.zeros_like(x, dtype=bool)
        kept_spans, dropped_spans = [], [(float(t[0]), float(t[-1]))] if len(t) else []
        x_clean = np.full_like(x, np.nan, dtype=float)
        _plot_debug(
            PLOT_DIR / f"{sid}_{wtag}_debug.png",
            sid,
            wtag,
            t,
            x,
            x_clean,
            keep_mask,
            None,
            spans,
            [],
            [],
        )
        collapse_frac_all = float(np.mean(_flat_runs(x, eps=FLAT_EQ_EPS, fs=fs)[2])) if len(x) else None
        collapse_max_run_all = int(_max_run_bool(_flat_runs(x, eps=FLAT_EQ_EPS, fs=fs)[2])) if len(x) else None
        # IMPORTANT: even for early-reject, we must overwrite the window values to NaN
        df_out = df.copy()
        df_out[vcol] = np.nan
        _write_resp_file(df_out, path)
        return RespCleanRow(
            subject_id=sid,
            final_order=int(wtag[1:]),
            file=path.name,
            status="reject",
            reasons=reasons,
            kept_spans=kept_spans,
            dropped_spans=dropped_spans,
            expected_duration_s=expected_dur,
            time_median_dt=float(median_dt) if np.isfinite(median_dt) else None,
            time_gap_frac=float(gap_frac) if np.isfinite(gap_frac) else None,
            time_median_dt_dev=float(median_dt_dev) if np.isfinite(median_dt_dev) else None,
            kept_frac=0.0,
            kept_dur_s=0.0,
            kept_cycles=0,
            br_bpm_median=None,
            period_cv=None,
            psd_peak_hz=None,
            psd_peak_ratio=None,
            psd_peak_prom=None,
            ptp_raw_p95p5=None,
            ptp_bp_p95p5=None,
            clip_frac_all=float(np.mean(_clip_mask(x))) if len(x) else None,
            clip_max_run_all=int(_max_run_bool(_clip_mask(x))) if len(x) else None,
            flat_max_run_all=int(_flat_runs(x, eps=FLAT_EQ_EPS, fs=fs)[1]) if len(x) else None,
            collapse_frac_all=collapse_frac_all,
            collapse_max_run_all=collapse_max_run_all,
            step2_checks=step2_checks,
        )

    # Step 2：在每个候选片段上判“像不像呼吸”
    candidates: List[Tuple[int, int, Dict[str, float], np.ndarray]] = []
    for (a, b) in spans:
        ta0 = t[a:b]
        xa0 = x[a:b]

        # Step2 前：先剔除长时间工程垃圾段，再把剩余连续片段切成 subspans
        subspans, keep_local = _pretrim_bad_runs_and_subspans(ta0, xa0, fs)
        for (sa, sb) in subspans:
            ga, gb = a + sa, a + sb
            ta = t[ga:gb]
            xa = x[ga:gb]

            ok, m2, xf = _is_physio_valid(ta, xa, fs)

            # 记录每次尝试的“可复核证据”
            rec: Dict[str, object] = {
                "t0": float(ta[0]) if len(ta) else np.nan,
                "t1": float(ta[-1]) if len(ta) else np.nan,
                "dur": float(m2.get("dur", np.nan)) if isinstance(m2, dict) else np.nan,
                "tag": str(m2.get("tag", "")) if isinstance(m2, dict) else "",
            }
            # 常用关键量（没有就留空/NaN）
            for k in [
                "ptp_raw_p95p5",
                "ptp_bp_p95p5",
                "psd_peak_hz",
                "psd_peak_ratio",
                "psd_peak_prom",
                "br_bpm",
                "br_bpm_psd",
                "period_cv",
                "n_cycles",
            ]:
                if isinstance(m2, dict) and k in m2:
                    rec[k] = m2.get(k)
            step2_checks.append(rec)

            # 让 debug 图在 reject 时也能画出带通波形（只要 bandpass 成功）
            tested_mask[ga:gb] = True
            if xf is not None and np.any(np.isfinite(xf)):
                xf_tested[ga:gb] = xf

            if ok:
                candidates.append((ga, gb, m2, xf))

    if len(candidates) == 0:
        reasons.append("no_physio_rhythm")
        keep_mask = np.zeros_like(x, dtype=bool)
        kept_spans, dropped_spans = [], [(float(t[0]), float(t[-1]))]
        x_clean = np.full_like(x, np.nan, dtype=float)
        _plot_debug(
            PLOT_DIR / f"{sid}_{wtag}_debug.png",
            sid,
            wtag,
            t,
            x,
            x_clean,
            keep_mask,
            xf_tested if np.any(np.isfinite(xf_tested)) else None,
            spans,
            [],
            [],
            band_mask=tested_mask,
        )
        collapse_frac_all = float(np.mean(_flat_runs(x, eps=FLAT_EQ_EPS, fs=fs)[2])) if len(x) else None
        collapse_max_run_all = int(_max_run_bool(_flat_runs(x, eps=FLAT_EQ_EPS, fs=fs)[2])) if len(x) else None
        # IMPORTANT: even for early-reject, we must overwrite the window values to NaN
        df_out = df.copy()
        df_out[vcol] = np.nan
        _write_resp_file(df_out, path)
        return RespCleanRow(
            subject_id=sid,
            final_order=int(wtag[1:]),
            file=path.name,
            status="reject",
            reasons=reasons,
            kept_spans=kept_spans,
            dropped_spans=dropped_spans,
            expected_duration_s=expected_dur,
            time_median_dt=float(median_dt) if np.isfinite(median_dt) else None,
            time_gap_frac=float(gap_frac) if np.isfinite(gap_frac) else None,
            time_median_dt_dev=float(median_dt_dev) if np.isfinite(median_dt_dev) else None,
            kept_frac=0.0,
            kept_dur_s=0.0,
            kept_cycles=0,
            br_bpm_median=None,
            period_cv=None,
            psd_peak_hz=None,
            psd_peak_ratio=None,
            psd_peak_prom=None,
            ptp_raw_p95p5=None,
            ptp_bp_p95p5=None,
            clip_frac_all=float(np.mean(_clip_mask(x))) if len(x) else None,
            clip_max_run_all=int(_max_run_bool(_clip_mask(x))) if len(x) else None,
            flat_max_run_all=int(_flat_runs(x, eps=FLAT_EQ_EPS, fs=fs)[1]) if len(x) else None,
            collapse_frac_all=collapse_frac_all,
            collapse_max_run_all=collapse_max_run_all,
            step2_checks=step2_checks,
        )

    # Step 3：平台峰规范化（不再切碎/删周期，只让平顶峰中心点唯一最高）
    keep_mask = np.zeros_like(x, dtype=bool)
    xf_keep = np.full_like(x, np.nan, dtype=float)
    x_mod = x.copy()

    plateau_spans_all: List[Tuple[float, float]] = []
    plateau_centers_all: List[float] = []

    # 先把 Step2 通过的片段并起来（multi 的基础）
    for (a, b, m2, xf_seg) in candidates:
        keep_mask[a:b] = True
        xf_keep[a:b] = xf_seg

        # 在该片段内做“平顶峰规范化”（仅改数值，不改 mask）
        try:
            x_fixed, psp, pct = _canonicalize_plateau_peaks_segment(t[a:b], x_mod[a:b], xf_seg, fs)
            x_mod[a:b] = x_fixed
            plateau_spans_all.extend(psp)
            plateau_centers_all.extend(pct)
        except Exception:
            pass

    # CLEAN_MODE=longest：只保留最长连续有效片段；multi：保留全部有效片段
    if CLEAN_MODE == "longest":
        kept_sp0, _ = _mask_to_spans(t, keep_mask)
        if kept_sp0:
            best = max(kept_sp0, key=lambda ab: ab[1] - ab[0])
            keep_mask = (t >= best[0]) & (t <= best[1]) & np.isfinite(x_mod)
        else:
            keep_mask = np.zeros_like(x, dtype=bool)

    # 最终清理结果：仅在 keep_mask 里保留（其余置 NaN）
    x_clean = x_mod.copy()
    x_clean[~keep_mask] = np.nan

    kept_spans, dropped_spans = _mask_to_spans(t, keep_mask)
    kept_dur = float(np.sum([te - ts for (ts, te) in kept_spans])) if kept_spans else 0.0
    kept_frac = float(np.mean(keep_mask)) if len(keep_mask) else 0.0

    # 用最终保留片段（cleaned）重新计算一次 Step2 指标，避免“报表与最终保留不一致”
    kept_metrics: List[Dict[str, float]] = []
    xf_for_plot = np.full_like(x, np.nan, dtype=float)

    for (ts, te) in kept_spans:
        seg = (t >= ts) & (t <= te) & np.isfinite(x_clean)
        if np.sum(seg) < 5:
            continue
        ta = t[seg]
        xa = x_clean[seg]
        ok2, m2b, xf2 = _is_physio_valid(ta, xa, fs)
        if ok2:
            kept_metrics.append(m2b)
            xf_for_plot[seg] = xf2

    if len(kept_metrics) == 0 or kept_dur < MIN_VALID_S:
        reasons.append("no_valid_span")
        status = "reject"
    else:
        status = "ok" if kept_frac >= 0.95 else "partial"
        if status == "partial":
            reasons.append("trimmed")

    # debug plot（始终生成，避免“黑箱”）
    _plot_debug(
        PLOT_DIR / f"{sid}_{wtag}_debug.png",
        sid,
        wtag,
        t,
        x,
        x_clean,
        keep_mask,
        xf_for_plot,
        spans,
        plateau_spans_all,
        plateau_centers_all,
        band_mask=keep_mask,
    )

    # 写回清理后的数据：x_clean 已经包含最终结果（multi/longest 已在上游处理）
    df_out = df.copy()
    if status == "reject":
        df_out[vcol] = np.nan
    else:
        df_out[vcol] = x_clean
    _write_resp_file(df_out, path)

    # 聚合最终指标（取中位数更稳健）
    def _median_key(key: str) -> Optional[float]:
        vals = [m.get(key, np.nan) for m in kept_metrics]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) == 0:
            return None
        return float(np.median(vals))

    br_bpm = _median_key("br_bpm")
    per_cv = _median_key("period_cv")
    psd_hz = _median_key("psd_peak_hz")
    psd_ratio = _median_key("psd_peak_ratio")
    psd_prom = _median_key("psd_peak_prom")
    ptp_raw = _median_key("ptp_raw_p95p5")
    ptp_bp = _median_key("ptp_bp_p95p5")
    cycles = _median_key("n_cycles")
    kept_cycles = int(round(cycles)) if cycles is not None else 0

    # 全窗工程证据
    clip = _clip_mask(x)
    flat_mask, flat_max_run, collapse_mask = _flat_runs(x, eps=FLAT_EQ_EPS, fs=fs)
    clip_frac_all = float(np.mean(clip)) if len(x) else None
    clip_max_run = int(_max_run_bool(clip)) if len(x) else None
    collapse_frac_all = float(np.mean(collapse_mask)) if len(x) else None
    collapse_max_run = int(_max_run_bool(collapse_mask)) if len(x) else None

    return RespCleanRow(
        subject_id=sid,
        final_order=int(wtag[1:]),
        file=path.name,
        status=status,
        reasons=reasons,
        kept_spans=kept_spans,
        dropped_spans=dropped_spans,
        expected_duration_s=expected_dur,
        time_median_dt=float(median_dt) if np.isfinite(median_dt) else None,
        time_gap_frac=float(gap_frac) if np.isfinite(gap_frac) else None,
        time_median_dt_dev=float(median_dt_dev) if np.isfinite(median_dt_dev) else None,
        kept_frac=float(kept_frac) if np.isfinite(kept_frac) else None,
        kept_dur_s=float(kept_dur) if np.isfinite(kept_dur) else None,
        kept_cycles=kept_cycles,
        br_bpm_median=br_bpm,
        period_cv=per_cv,
        psd_peak_hz=psd_hz,
        psd_peak_ratio=psd_ratio,
        psd_peak_prom=psd_prom,
        ptp_raw_p95p5=ptp_raw,
        ptp_bp_p95p5=ptp_bp,
        clip_frac_all=clip_frac_all,
        clip_max_run_all=clip_max_run,
        flat_max_run_all=int(flat_max_run) if np.isfinite(flat_max_run) else None,
        collapse_frac_all=collapse_frac_all,
        collapse_max_run_all=collapse_max_run,
        step2_checks=step2_checks,
    )


# ------------------------------
# main
# ------------------------------

def main() -> None:
    print("[clean_resp] SRC_DIR=", SRC_DIR)
    print("[clean_resp] OUT_ROOT=", OUT_ROOT)
    print("[clean_resp] windowing_report=", windowing_report)
    print("[clean_resp] REPORT_DIR=", REPORT_DIR)
    print("[clean_resp] PLOT_DIR=", PLOT_DIR)
    print("[clean_resp] CLEAN_MODE=", CLEAN_MODE)
    print("[clean_resp] FS_DEFAULT=", FS_DEFAULT, "ADC=", (ADC_MIN, ADC_MAX), "EPS=", ADC_EPS)
    print("[clean_resp] PRETRIM_USE_CLIP=", PRETRIM_USE_CLIP, "PRETRIM_MIN_CLIP_RUN_S=", PRETRIM_MIN_CLIP_RUN_S)

    rep = _load_windowing_report()

    # 建立 (sid, final_order)->expected_duration 的映射（如果报表里没有 duration 列就 None）
    dur_map: Dict[Tuple[str, int], Optional[float]] = {}
    if w_duration_col in rep.columns:
        for _, r in rep.iterrows():
            dur_map[(str(r["subject_id"]), int(r[w_id_col]))] = float(r[w_duration_col])
    else:
        for _, r in rep.iterrows():
            dur_map[(str(r["subject_id"]), int(r[w_id_col]))] = None

    files = _list_window_files()

    # 过滤 SID/WIN
    sid_set = set(SID) if len(SID) else None
    win_set = set(WIN) if len(WIN) else None

    todo: List[Tuple[Path, str, str, Optional[float]]] = []
    for p in files:
        sid, wtag = _parse_sid_wid_from_name(p.name)
        if sid_set is not None and sid not in sid_set:
            continue
        if win_set is not None and wtag not in win_set:
            continue
        wnum = int(wtag[1:])
        expected = dur_map.get((sid, wnum), None)
        todo.append((p, sid, wtag, expected))

    print(f"[clean_resp] to_process={len(todo)} windows")

    rows: List[RespCleanRow] = []
    for (p, sid, wtag, expected) in tqdm(todo, desc="clean_resp", unit="win"):
        try:
            row = _process_one_window(p, expected, sid, wtag)
        except Exception as e:
            # 出错也要留证据，避免静默失败
            print(f"[clean_resp][ERROR] {p.name}: {e}")
            row = RespCleanRow(
                subject_id=sid,
                final_order=int(wtag[1:]),
                file=p.name,
                status="reject",
                reasons=["exception"],
                kept_spans=[],
                dropped_spans=[],
                expected_duration_s=expected,
                time_median_dt=None,
                time_gap_frac=None,
                time_median_dt_dev=None,
                kept_frac=0.0,
                kept_dur_s=0.0,
                kept_cycles=0,
                br_bpm_median=None,
                period_cv=None,
                psd_peak_hz=None,
                psd_peak_ratio=None,
                psd_peak_prom=None,
                ptp_raw_p95p5=None,
                ptp_bp_p95p5=None,
                clip_frac_all=None,
                clip_max_run_all=None,
                flat_max_run_all=None,
                collapse_frac_all=None,
                collapse_max_run_all=None,
                step2_checks=[],
            )
        rows.append(row)

    # 输出汇总报告
    out_csv = REPORT_DIR / "clean_resp_summary.csv"
    df_report = pd.DataFrame([asdict(r) for r in rows])
    df_report.to_csv(out_csv, index=False)

    # 简要统计
    vc = df_report["status"].value_counts(dropna=False).to_dict()
    print("[clean_resp] status_counts=", vc)
    print("[clean_resp] wrote:")
    print("  -", out_csv)
    print("  - debug plots:", PLOT_DIR)
    print("  - cleaned resp files overwritten in:", OUT_ROOT)

    # 给用户一眼能看懂的“列解释提示”
    print("[clean_resp] report columns (human):")
    print("  status: ok/partial/reject")
    print("  kept_spans/dropped_spans: 保留/剔除的时间区间(秒)")
    print("  time_*: Step1 工程底线摘要")
    print("  br_bpm_median/period_cv/psd_*: Step2 生理节律摘要")
    print("  ptp_raw_p95p5/ptp_bp_p95p5/psd_peak_prom: Step2 的‘幅度门槛/谱峰突出度’证据（用于拦截伪呼吸）")
    print("  clip_*/flat_*/collapse_*: Step3 工程证据(全窗统计, 图中有标注)")


if __name__ == "__main__":
    main()