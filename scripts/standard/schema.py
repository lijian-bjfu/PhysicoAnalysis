# scripts/standard/schema.py
import numpy as np
import pandas as pd

def _is_mono_increasing(x: pd.Series) -> bool:
    if x.isna().any(): return False
    dx = np.diff(x.to_numpy(dtype=float))
    return np.all(dx > 0)

def to_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """校验并返回连续信号标准格式：['time_s','value','fs_hz']"""
    need = {"time_s","value"}
    if not need.issubset(df.columns):
        raise ValueError(f"连续信号缺列：需要 {need}")
    out = df[["time_s","value"]].copy()
    # time_s 单调递增
    out = out.dropna().sort_values("time_s")
    if len(out) < 3:
        raise ValueError("连续信号太短")
    if not _is_mono_increasing(out["time_s"]):
        # 去重再试
        out = out[~out["time_s"].diff().fillna(1).eq(0)]
        if not _is_mono_increasing(out["time_s"]):
            raise ValueError("time_s 不是严格递增")

    # fs_hz：优先保留，缺则估计
    if "fs_hz" in df.columns and pd.notna(df["fs_hz"].iloc[0]):
        fs = float(df["fs_hz"].iloc[0])
    else:
        dt = out["time_s"].diff().median()
        fs = float(1.0/dt) if pd.notna(dt) and dt>0 else np.nan
    out["fs_hz"] = fs if np.isfinite(fs) else np.nan
    return out.reset_index(drop=True)

def to_rr(df: pd.DataFrame) -> pd.DataFrame:
    """校验并返回逐搏信号标准格式：['t_s','rr_ms']"""
    need = {"t_s","rr_ms"}
    # 允许把你那种 time_lsl + ms 映射进来后，统一叫 t_s, rr_ms
    if not need.issubset(df.columns):
        raise ValueError(f"RR 缺列：需要 {need}")
    out = df[["t_s","rr_ms"]].copy().dropna().sort_values("t_s")
    if len(out) < 3:
        raise ValueError("RR 序列太短")
    if not _is_mono_increasing(out["t_s"]):
        out = out[~out["t_s"].diff().fillna(1).eq(0)]
        if not _is_mono_increasing(out["t_s"]):
            raise ValueError("t_s 不是严格递增")
    return out.reset_index(drop=True)

def to_hr(df: pd.DataFrame) -> pd.DataFrame:
    """
    规范化心率序列（HR）到统一 schema：
      输入（来自 relabel.map_to_standard 后）需包含列：
        - time_s : float，统一的时间轴（秒，可来自 LSL 或换算）
        - hr_bpm : float，心率（次/分）
      输出：
        - DataFrame[["time_s","hr_bpm"]]，按 time_s 排序，去除 NaN 与重复时间戳（保留首个）
    本函数不做重采样、不做阈值裁剪，只做最小必要清洗，以免“规范化阶段”污染数据。
    """
    need = {"time_s", "hr_bpm"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"to_hr(): HR 缺列：需要 {need}，缺少 {missing}")

    out = df[["time_s", "hr_bpm"]].copy()

    # 数值化 + 去 NaN
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
    out["hr_bpm"] = pd.to_numeric(out["hr_bpm"], errors="coerce")
    out = out.dropna(subset=["time_s", "hr_bpm"])

    # 排序 + 去重（相同 time_s 只保留首个）
    out = out.sort_values("time_s")
    out = out.loc[~out["time_s"].duplicated(keep="first")]

    # 重置行索引
    return out.reset_index(drop=True)

def to_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    规范化事件标注（HR）到统一 schema：
      输入（来自 relabel.map_to_standard 后）需包含列：
        - time_s : float，统一的时间轴（秒，可来自 LSL 或换算）
        - events : str, 如"baseline_start","stim_start",stim_end"等等
      输出：
        - DataFrame[["time_s","events"]]，按 time_s 排序
    """
    need = {"time_s", "events"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"to_events(): HR 缺列：需要 {need}，缺少 {missing}")

    out = df[["time_s", "events"]].copy()

    # 事件名统一为字符串并去首尾空白
    out["events"] = out["events"].astype(str).str.strip()
    # 丢弃缺失与空字符串
    out = out.dropna(subset=["time_s"]) \
             .loc[out["events"].ne("")]

    # 重置行索引
    return out.reset_index(drop=True)

def to_acc(df: pd.DataFrame) -> pd.DataFrame:
    """
    规范化事件标注（HR）到统一 schema：
      输入（来自 relabel.map_to_standard 后）需包含列：
        - time_s  : float，统一时间轴（秒）
        - value_x : float，加速度 X 轴（单位保持与原始一致，如 mG）
        - value_y : float，加速度 Y 轴
        - value_z : float，加速度 Z 轴
      输出：
        - DataFrame[["time_s","value_x","value_y","value_z"]]
    """
    need = {"time_s","value_x","value_y","value_z"}
     
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"to_events(): HR 缺列：需要 {need}，缺少 {missing}")

    out = df[["time_s", "value_x", "value_y", "value_z"]].copy()
        # 数值化
    out["time_s"]  = pd.to_numeric(out["time_s"], errors="coerce")
    out["value_x"] = pd.to_numeric(out["value_x"], errors="coerce")
    out["value_y"] = pd.to_numeric(out["value_y"], errors="coerce")
    out["value_z"] = pd.to_numeric(out["value_z"], errors="coerce")

    return out.reset_index(drop=True)