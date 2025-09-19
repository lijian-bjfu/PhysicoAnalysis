# scripts/standard/relabel.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

def _as_seconds(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s = s[np.isfinite(s)]
    if s.size < 3:
        return s

    # 典型步长（相邻差的中位数的绝对值）
    d = np.diff(s)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return s

    dt = float(np.median(np.abs(d)))

    # 数值量级（用于区分“很大的秒级绝对时间戳”与“毫秒/微秒刻度”）
    med_val = float(np.median(s))
    max_val = float(np.max(s))

    # 规则说明：
    # - 微秒：相邻步长非常大（>= 1e5），且通常数值整体也很大 → 除以 1e6
    # - 毫秒：相邻步长在 [10, 1e5) 之间，但仅当时间戳的量级不巨大时才认为是毫秒。
    #         若时间戳本身已在 1e5 以上（如 LSL 提供的“秒级绝对时间” 7e5），
    #         即使 dt > 10 也按“秒”处理，避免把稀疏事件误判为毫秒。
    # - 其余：按“秒”返回。

    if dt >= 100000:  # 微秒
        return s / 1e6

    if 10 <= dt < 100000:
        if max_val < 1e5 and med_val < 1e5:
            # 真·毫秒时间戳（数值量级不大，步长却在几十/几百量级）
            return s / 1000.0
        # 稀疏事件但为“秒级绝对时间”，保持不动
        return s

    # 默认：已经是秒
    return s

def _first_two_numeric(df: pd.DataFrame) -> tuple[str, str]:
    """找前两列数值列，第一列默认为时间，第二列默认为值。"""
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(nums) >= 2:
        return nums[0], nums[1]
    cols = list(df.columns)
    return cols[0], (cols[1] if len(cols) > 1 else cols[0])

def map_to_standard(sig: str, df: pd.DataFrame, options: Optional[dict] = None) -> pd.DataFrame:
    """
    把各种来源的原始表映射为统一规范。
      - sig: 信号名。多来自数据名后缀
      - df: 数据列表，提供列名 
    约定输出：
      - ECG/RESP/PPG：["time_s","value", 可选 "fs_hz"]
      - RR/PPI：     ["t_s","rr_ms"]
      - HR：         ["time_s","hr_bpm"]
      - ACC：        ["time_s", ("acc_mag" 或 "value_x","value_y","value_z"), 可选 "fs_hz"]
      - EVENTS：     ["time_s","events", 可选 "duration_s"]
    options 可留空；后续你要细化口径（比如强制单位、列别名），可往里塞配置。
    """
    dfl = df.copy()
    cmap = {c.lower(): c for c in dfl.columns}

    # 默认前两列分别为时间与信号值
    time, value = _first_two_numeric(dfl)

    # --------- ECG ----------
    if sig == "ecg":
        # 优先常见列名；否则按列序兜底
        out = pd.DataFrame({
            "time_s": _as_seconds(dfl[time]),
            "value":  pd.to_numeric(dfl[value], errors="coerce")
        }).dropna()
        # 采样率估计（如无）
        if "fs_hz" in dfl.columns:
            out["fs_hz"] = pd.to_numeric(dfl["fs_hz"], errors="coerce").iloc[0]
        return out

    # --------- RR/PPI ----------
    if sig in ("rr", "ppi"):
        t = _as_seconds(dfl[time])
        rr = pd.to_numeric(dfl[value], errors="coerce")
        # 单位归一：如果 rr 像“秒”（< 10），转毫秒
        rr_ms = np.where(rr < 10.0, rr * 1000.0, rr)
        out = pd.DataFrame({"t_s": t, "rr_ms": rr_ms}).dropna()
        return out

    # --------- HR ----------
    if sig == "hr":
        out = pd.DataFrame({
            "time_s": _as_seconds(dfl[time]),
            "hr_bpm": pd.to_numeric(dfl[value], errors="coerce")
        })
        return out.dropna(subset=["time_s","hr_bpm"])

    # --------- RESP ----------
    if sig == "resp":
        out = pd.DataFrame({
            "time_s": _as_seconds(dfl[time]),
            "value":  pd.to_numeric(dfl[value], errors="coerce")
        }).dropna()
        if "fs_hz" in dfl.columns:
            out["fs_hz"] = pd.to_numeric(dfl["fs_hz"], errors="coerce").iloc[0]
        return out

    # --------- PPG ----------
    if sig == "ppg":
        out = pd.DataFrame({
            "time_s": _as_seconds(dfl[time]),
            "value":  pd.to_numeric(dfl[value], errors="coerce")
        }).dropna()
        if "fs_hz" in dfl.columns:
            out["fs_hz"] = pd.to_numeric(dfl["fs_hz"], errors="coerce").iloc[0]
        return out

    # --------- ACC ----------
    # 仅支持 local
    if sig == "acc":
        vx = cmap.get("x_mg")
        vy = cmap.get("y_mg")
        vz = cmap.get("z_mg")

        base = {"time_s": _as_seconds(dfl[time])}

        if not (vx and vy and vz):
            cands = [c for c in dfl.columns if c != time][:3]
            if len(cands) == 3:
                vx, vy, vz = cands
            if vx: base["value_x"] = pd.to_numeric(dfl[vx], errors="coerce")
            if vy: base["value_y"] = pd.to_numeric(dfl[vy], errors="coerce")
            if vz: base["value_z"] = pd.to_numeric(dfl[vz], errors="coerce")

        out = pd.DataFrame(base).dropna()
        if "fs_hz" in dfl.columns:
            out["fs_hz"] = pd.to_numeric(dfl["fs_hz"], errors="coerce").iloc[0]
        return out

    # --------- EVENTS / MARKERS ----------
    # 仅支持 local
    if sig in ("markers", "events"):
        out = pd.DataFrame({
            "time_s": _as_seconds(dfl[time]),
            "events":  dfl[value]
        }).dropna()
        return out

    # 未识别
    raise ValueError(f"map_to_standard: 未支持的信号类型 '{sig}'")