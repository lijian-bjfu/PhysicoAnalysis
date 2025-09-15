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

    d = np.diff(s)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return s

    dt = float(np.median(np.abs(d)))

    # 判定规则：毫秒 / 微秒 / 否则按秒
    if 10 <= dt < 100000:       # 相邻步长 >= 10（比如 800、1000），很像“毫秒”
        return s / 1000.0
    if dt >= 100000:            # 相邻步长巨大，像“微秒”
        return s / 1e6

    # 默认：已经是“秒”（包括 LSL 秒、WESAD 相对秒、连续采样秒）
    return s

def _first_two_numeric(df: pd.DataFrame) -> tuple[str, str]:
    """找前两列数值列，兜底用。"""
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(nums) >= 2:
        return nums[0], nums[1]
    cols = list(df.columns)
    return cols[0], (cols[1] if len(cols) > 1 else cols[0])

def map_to_standard(sig: str, df: pd.DataFrame, options: Optional[dict] = None) -> pd.DataFrame:
    """
    把各种来源的原始表映射为统一规范。
    约定输出：
      - ECG/RESP/PPG：["time_s","value", 可选 "fs_hz"]
      - RR/PPI：     ["t_s","rr_ms"]
      - HR：         ["time_s","hr_bpm"]
      - ACC：        ["time_s", ("acc_mag" 或 "value_x","value_y","value_z"), 可选 "fs_hz"]
      - EVENTS：     ["t_s","label", 可选 "duration_s"]
    options 可留空；后续你要细化口径（比如强制单位、列别名），可往里塞配置。
    """
    opts = options or {}
    dfl = df.copy()
    cmap = {c.lower(): c for c in dfl.columns}

    # --------- ECG ----------
    if sig == "ecg":
        # 优先常见列名；否则按列序兜底
        tcol = cmap.get("time_s") or cmap.get("time") or cmap.get("timestamp") or cmap.get("time_lsl")
        vcol = cmap.get("value")  or cmap.get("ecg") or cmap.get("uv") or cmap.get("amplitude")
        if tcol is None or vcol is None:
            tcol2, vcol2 = _first_two_numeric(dfl)
            tcol = tcol or tcol2
            vcol = vcol or vcol2
        out = pd.DataFrame({
            "time_s": _as_seconds(dfl[tcol]),
            "value":  pd.to_numeric(dfl[vcol], errors="coerce")
        }).dropna()
        # 采样率估计（如无）
        if "fs_hz" in dfl.columns:
            out["fs_hz"] = pd.to_numeric(dfl["fs_hz"], errors="coerce").iloc[0]
        else:
            dt = out["time_s"].diff().median()
            out["fs_hz"] = float(round(1.0 / dt)) if pd.notna(dt) and dt > 0 else np.nan
        return out

    # --------- RR/PPI ----------
    if sig in ("rr", "ppi"):
        # 时间列可能叫 t_s / time_s / time / time_lsl
        tcol = cmap.get("t_s") or cmap.get("time_s") or cmap.get("time") or cmap.get("time_lsl") or cmap.get("timestamp")
        rcol = cmap.get("rr_ms") or cmap.get("ppi_ms") or cmap.get("rr") or cmap.get("ibi")
        if tcol is None or rcol is None:
            tcol2, rcol2 = _first_two_numeric(dfl)
            tcol = tcol or tcol2
            rcol = rcol or rcol2
        t = _as_seconds(dfl[tcol])
        rr = pd.to_numeric(dfl[rcol], errors="coerce")
        # 单位归一：如果 rr 像“秒”（< 10），转毫秒
        rr_ms = np.where(rr < 10.0, rr * 1000.0, rr)
        out = pd.DataFrame({"t_s": t, "rr_ms": rr_ms}).dropna()
        return out

    # --------- HR ----------
    if sig == "hr":
        # 心率（每秒或不规则采样都可）
        tcol = cmap.get("time_s") or cmap.get("time_lsl") or cmap.get("timestamp") or cmap.get("time")
        hcol = cmap.get("hr_bpm") or cmap.get("hr") or cmap.get("bpm")
        if tcol is None or hcol is None:
            tcol2, hcol2 = _first_two_numeric(dfl)
            tcol = tcol or tcol2
            hcol = hcol or hcol2
        out = pd.DataFrame({
            "time_s": _as_seconds(dfl[tcol]),
            "hr_bpm": pd.to_numeric(dfl[hcol], errors="coerce")
        })
        return out.dropna(subset=["time_s","hr_bpm"])

    # --------- RESP ----------
    if sig == "resp":
        tcol = cmap.get("time_s") or cmap.get("time") or cmap.get("timestamp") or cmap.get("time_lsl")
        vcol = cmap.get("value")  or cmap.get("resp") or cmap.get("amplitude")
        if tcol is None or vcol is None:
            tcol2, vcol2 = _first_two_numeric(dfl)
            tcol = tcol or tcol2
            vcol = vcol or vcol2
        out = pd.DataFrame({
            "time_s": _as_seconds(dfl[tcol]),
            "value":  pd.to_numeric(dfl[vcol], errors="coerce")
        }).dropna()
        if "fs_hz" in dfl.columns:
            out["fs_hz"] = pd.to_numeric(dfl["fs_hz"], errors="coerce").iloc[0]
        else:
            dt = out["time_s"].diff().median()
            out["fs_hz"] = float(round(1.0 / dt)) if pd.notna(dt) and dt > 0 else np.nan
        return out

    # --------- PPG ----------
    if sig == "ppg":
        tcol = cmap.get("time_s") or cmap.get("time") or cmap.get("timestamp") or cmap.get("time_lsl")
        vcol = cmap.get("value")  or cmap.get("ppg") or cmap.get("bvp") or cmap.get("amplitude")
        if tcol is None or vcol is None:
            tcol2, vcol2 = _first_two_numeric(dfl)
            tcol = tcol or tcol2
            vcol = vcol or vcol2
        out = pd.DataFrame({
            "time_s": _as_seconds(dfl[tcol]),
            "value":  pd.to_numeric(dfl[vcol], errors="coerce")
        }).dropna()
        if "fs_hz" in dfl.columns:
            out["fs_hz"] = pd.to_numeric(dfl["fs_hz"], errors="coerce").iloc[0]
        else:
            dt = out["time_s"].diff().median()
            out["fs_hz"] = float(round(1.0 / dt)) if pd.notna(dt) and dt > 0 else np.nan
        return out

    # --------- ACC ----------
    if sig == "acc":
        tcol = cmap.get("time_s") or cmap.get("time") or cmap.get("timestamp") or cmap.get("time_lsl")
        vm = cmap.get("acc_mag") or cmap.get("mag")
        vx = cmap.get("value_x") or cmap.get("ax") or cmap.get("acc_x")
        vy = cmap.get("value_y") or cmap.get("ay") or cmap.get("acc_y")
        vz = cmap.get("value_z") or cmap.get("az") or cmap.get("acc_z")
        if tcol is None:
            tcol = dfl.columns[0]
        base = {"time_s": _as_seconds(dfl[tcol])}
        if vm:
            base["acc_mag"] = pd.to_numeric(dfl[vm], errors="coerce")
        else:
            if not (vx and vy and vz):
                cands = [c for c in dfl.columns if c != tcol][:3]
                if len(cands) == 3:
                    vx, vy, vz = cands
            if vx: base["value_x"] = pd.to_numeric(dfl[vx], errors="coerce")
            if vy: base["value_y"] = pd.to_numeric(dfl[vy], errors="coerce")
            if vz: base["value_z"] = pd.to_numeric(dfl[vz], errors="coerce")
        out = pd.DataFrame(base).dropna()
        if "fs_hz" in dfl.columns:
            out["fs_hz"] = pd.to_numeric(dfl["fs_hz"], errors="coerce").iloc[0]
        else:
            dt = out["time_s"].diff().median()
            out["fs_hz"] = float(round(1.0 / dt)) if pd.notna(dt) and dt > 0 else np.nan
        return out

    # --------- EVENTS / MARKERS ----------
    if sig in ("markers", "events"):
        # 兼容 BIDS: onset, duration, trial_type
        onset = cmap.get("t_s") or cmap.get("time_s") or cmap.get("onset") or cmap.get("time") or cmap.get("timestamp") or cmap.get("time_lsl")
        label = cmap.get("label") or cmap.get("event") or cmap.get("trial_type") or cmap.get("name")
        dur   = cmap.get("duration_s") or cmap.get("duration") or cmap.get("dur")
        if onset is None:
            # 兜底用第一列
            onset = dfl.columns[0]
        out = pd.DataFrame({"t_s": _as_seconds(dfl[onset])})
        if label:
            out["label"] = dfl[label].astype(str)
        if dur:
            dur_s = pd.to_numeric(dfl[dur], errors="coerce")
            # 若像“毫秒”，转秒
            out["duration_s"] = np.where(dur_s > 50, dur_s/1000.0, dur_s)
        return out.dropna(subset=["t_s"])

    # 未识别
    raise ValueError(f"map_to_standard: 未支持的信号类型 '{sig}'")