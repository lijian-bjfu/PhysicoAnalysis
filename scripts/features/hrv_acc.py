import pandas as pd
import numpy as np
from typing import Optional

import sys
from pathlib import Path

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---


def _stdcol_acc(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """标准化加速度列名为 ['t_s','ax','ay','az']；若列不足返回 None。"""
    if df is None or df.empty:
        return None
    acc = df.copy()
    # 时间列：既支持 't_s' 也支持 'time_s'
    if 't_s' in acc.columns:
        pass
    elif 'time_s' in acc.columns:
        acc = acc.rename(columns={'time_s': 't_s'})
    else:
        return None
    # 三轴列
    if {'ax', 'ay', 'az'}.issubset(acc.columns):
        cols = ['ax', 'ay', 'az']
    elif {'value_x', 'value_y', 'value_z'}.issubset(acc.columns):
        acc = acc.rename(columns={'value_x': 'ax', 'value_y': 'ay', 'value_z': 'az'})
        cols = ['ax', 'ay', 'az']
    else:
        return None
    return acc[['t_s'] + cols].astype(float)


# ---- Per-session baseline gravity (g0) cache ----
# Keyed by sid (session id like P001S001T001R001). Value is g0 in the same unit as vm (mG).
_ACC_G0_CACHE: dict[str, float] = {}
_WARNED_MISSING_BASELINE: set[str] = set()


def _estimate_g0_from_vm(vm: np.ndarray) -> float:
    """Robust g0 estimation.

    We estimate the gravity baseline from the lower-activity portion of vm to avoid
    over-estimating g0 in windows with sustained motion.

    Strategy:
      - take the 20th percentile cutoff q20
      - compute median(vm[vm <= q20])
      - if the subset is too small, fall back to median(vm)
    """
    vm = vm[np.isfinite(vm)]
    if vm.size == 0:
        return np.nan
    q20 = float(np.quantile(vm, 0.20))
    base = vm[vm <= q20]
    if base.size < max(10, int(0.05 * vm.size)):
        base = vm
    return float(np.median(base))


def _is_baseline_window(src: pd.DataFrame, wid: Optional[int]) -> bool:
    """Detect whether this window is baseline.

    Primary rule: wid == 1 (your pipeline's baseline window).
    Secondary rules: meaning/level contains 'baseline' (case-insensitive).
    """
    if wid == 1:
        return True
    for col in ("meaning", "level"):
        if col in src.columns and len(src[col]) > 0:
            s = str(src[col].iloc[0]).lower()
            if "baseline" in s:
                return True
    return False


def features_segment(
    rr_df: Optional[pd.DataFrame] = None,
    resp_df: Optional[pd.DataFrame] = None,
    *,
    acc_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    计算单窗口加速度**一项协变量**（用于统计模型的时间变化协变量）。
    为兼容主流程保留统一接口：
      - 推荐：传 acc_df=...
      - 兼容：传 resp_df=...（旧统一接口的第二参数）或 rr_df=...

    输入列（任选其一）：
      - time_s/t_s, value_x/value_y/value_z  或  t_s, ax/ay/az

    输出列（仅一项）：
      - acc_enmo_mean    : ENMO 均值，ENMO = max(||a|| - g0, 0)

    g0 估计：
      - g0 为重力基线，单位与数据一致。
      - 从同一会话的基线窗口（wid==1）中估计并缓存。
      - 若基线窗口缺失，则回退到当前窗口内的稳健估计。
    """
    # 解析数据来源：优先 acc_df，其次 resp_df，最后 rr_df（为兼容旧调用）
    src = acc_df if acc_df is not None else (resp_df if resp_df is not None else rr_df)
    acc = _stdcol_acc(src)

    if acc is None or acc.empty:
        return pd.DataFrame([{
            'acc_enmo_mean': np.nan,
        }])

    sid = None
    wid = None
    if src is not None:
        sid = src.attrs.get("sid") if hasattr(src, "attrs") else None
        wid = src.attrs.get("w_id") if hasattr(src, "attrs") else None
        try:
            wid = int(wid) if wid is not None else None
        except Exception:
            wid = None

    ax = acc['ax'].to_numpy()
    ay = acc['ay'].to_numpy()
    az = acc['az'].to_numpy()

    # 向量模
    vm = np.sqrt(ax*ax + ay*ay + az*az)

    # ---- g0 estimation (session baseline cached) ----
    if sid is not None and _is_baseline_window(src, wid):
        _ACC_G0_CACHE[sid] = _estimate_g0_from_vm(vm)

    if sid is not None and sid in _ACC_G0_CACHE and np.isfinite(_ACC_G0_CACHE[sid]):
        g0 = _ACC_G0_CACHE[sid]
    else:
        # Fallback: robust within-window estimation
        g0 = _estimate_g0_from_vm(vm)
        if sid is not None and sid not in _WARNED_MISSING_BASELINE:
            print(f"[WARN] ACC baseline g0 missing for sid={sid}; falling back to within-window g0.")
            _WARNED_MISSING_BASELINE.add(sid)

    # Vector magnitude and ENMO
    enmo = np.maximum(vm - g0, 0.0)

    out = {
        'acc_enmo_mean': float(np.nanmean(enmo)) if enmo.size else np.nan,
    }
    return pd.DataFrame([out])


def features_segment_acc(acc_df: pd.DataFrame) -> pd.DataFrame:
    """语义明确的便捷别名。等价于 features_segment(acc_df=acc_df)。"""
    return features_segment(acc_df=acc_df)


if __name__ == '__main__':
    print('features_segment(acc_df=...) -> DataFrame[1x1] (acc_enmo_mean)')