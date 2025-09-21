# scripts/standard/schema.py
import numpy as np
import pandas as pd

import sys
from pathlib import Path
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
from settings import SCHEMA

def _is_mono_increasing(x: pd.Series) -> bool:
    if x.isna().any(): return False
    dx = np.diff(x.to_numpy(dtype=float))
    return np.all(dx > 0)

def _first_present(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(f"缺少必要列：期望其一 {names}，但未找到。现有列：{list(df.columns)}")

def to_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """校验并返回连续信号标准格式：['time_s','value','fs_hz']"""
    need_time = [SCHEMA["ecg"]["t"], SCHEMA.get("ppg", {}).get("t", "t_s"), SCHEMA.get("resp", {}).get("t", "time_s"), "time_s", "t_s"]
    need_val  = [SCHEMA["ecg"]["v"],  SCHEMA.get("ppg", {}).get("v", "value"),   SCHEMA.get("resp", {}).get("v", "value"),   "value"]

    t_col = _first_present(df, need_time)
    v_col = _first_present(df, need_val)

    out = df[[t_col, v_col]].copy()
    # 清洗与排序
    out = out.dropna().sort_values(t_col)
    if len(out) < 3:
        raise ValueError("连续信号太短")
    if not _is_mono_increasing(out[t_col]):
        out = out[~out["time_s"].diff().fillna(1).eq(0)]
        if not _is_mono_increasing(out["time_s"]):
            raise ValueError(f"{t_col} 不是严格递增")

    # fs_hz：优先读 canonical 键，其次读通用 'fs_hz'，否则估计
    fs_key_candidates = [SCHEMA.get("ecg", {}).get("fs", "fs_hz"), SCHEMA.get("ppg", {}).get("fs", "fs_hz"), "fs_hz"]
    fs_key = None
    for k in fs_key_candidates:
        if k in df.columns:
            fs_key = k
            break

    if fs_key is not None and pd.notna(df[fs_key].iloc[0]):
        fs = float(df[fs_key].iloc[0])
    else:
        dt = out[t_col].diff().median()
        fs = float(1.0/dt) if pd.notna(dt) and dt>0 else np.nan
    out[fs_key] = fs if np.isfinite(fs) else np.nan
    return out.reset_index(drop=True)

def to_rr(df: pd.DataFrame) -> pd.DataFrame:
    """校验并返回逐搏信号标准格式：['t_s','rr_ms']"""
    t_key = SCHEMA["rr"]["t"]
    v_key = SCHEMA["rr"]["v"]
    need = {t_key, v_key}
    if not need.issubset(df.columns):
        raise ValueError(f"RR 缺列：需要 {need}")
    out = df[[t_key, v_key]].copy().dropna().sort_values(t_key)
    if len(out) < 3:
        raise ValueError("RR 序列太短")
    if not _is_mono_increasing(out[t_key]):
        out = out[~out[t_key].diff().fillna(1).eq(0)]
        if not _is_mono_increasing(out[t_key]):
            raise ValueError(f"{t_key} 不是严格递增")
    return out.reset_index(drop=True)

def to_hr(df: pd.DataFrame) -> pd.DataFrame:
    """
    规范化心率序列（HR）到统一 schema：
      输入（来自 relabel.map_to_standard 后）需包含列：
        - time_s : float，统一的时间轴（秒，可来自 LSL 或换算）
        - bpm : float，心率（次/分）
      输出：
        - DataFrame[["time_s","hr_bpm"]]，按 time_s 排序，去除 NaN 与重复时间戳（保留首个）
    """
    t_key = SCHEMA["hr"]["t"]
    v_key = SCHEMA["hr"]["v"]
    need = {t_key, v_key}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"to_hr(): HR 缺列：需要 {need}，缺少 {missing}")

    out = df[[t_key, v_key]].copy()
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
    t_key = SCHEMA["events"]["t"]
    l_key = SCHEMA["events"]["label"]
    need = {t_key, l_key}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"to_events(): 事件缺列：需要 {need}，缺少 {missing}")

    out = df[[t_key, l_key]].copy()
    out[l_key] = out[l_key].astype(str).str.strip()
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
    t_key  = SCHEMA["acc"]["t"]
    x_key  = SCHEMA["acc"]["vx"]
    y_key  = SCHEMA["acc"]["vy"]
    z_key  = SCHEMA["acc"]["vz"]
    need = {t_key, x_key, y_key, z_key}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"to_acc(): 加速度缺列：需要 {need}，缺少 {missing}")

    out = df[[t_key, x_key, y_key, z_key]].copy()
    out[t_key]  = pd.to_numeric(out[t_key], errors="coerce")
    out[x_key] = pd.to_numeric(out[x_key], errors="coerce")
    out[y_key] = pd.to_numeric(out[y_key], errors="coerce")
    out[z_key] = pd.to_numeric(out[z_key], errors="coerce")
    out = out.dropna(subset=[t_key]).sort_values(t_key)
    return out.reset_index(drop=True)