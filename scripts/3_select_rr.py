# scripts/2a_select_rr.py
import numpy as np, pandas as pd, sys
import matplotlib.pyplot as plt
# optional progress bar for long runs
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
from pathlib import Path

# boot
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
from settings import DATASETS, ACTIVE_DATA, DATA_DIR, PROCESSED_DIR, RR_COMPARE, PARAMS, RR_SCORE, sid_filter  # RR_SCORE 可选，若未在 settings 中定义将使用默认权重

paths = DATASETS[ACTIVE_DATA]["paths"]
SRC_NORM_DIR = (DATA_DIR / paths["norm"]).resolve()
RR_OUT_DIR   = (DATA_DIR / paths["confirmed"]).resolve()
PREVIEW_DIR  = (PROCESSED_DIR / "rr_select" / ACTIVE_DATA).resolve()
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# 画图横轴范围策略，基于事件标注起始点绘制，基于hr波长起始点
# 'auto' | 'events' | 'hr'
PLOT_RANGE_MODE = 'auto'


DEC_SUGG = PREVIEW_DIR / "decision_suggested.csv"
DEC_FILE = PREVIEW_DIR / "decision.csv"

# 读取 ecg 数据
def _read_ecg(sid: str):
    p = SRC_NORM_DIR / f"{sid}_ecg.parquet"
    if not p.exists(): p = SRC_NORM_DIR / f"{sid}_ecg.csv"
    if not p.exists(): return None
    df = pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
    return df

# 用 NeuroKit2 在连续心电上找 R 峰，然后转成逐搏 RR 表（两列：t_s 秒、rr_ms 毫秒）
def _ecg_to_rr(ecg_df: pd.DataFrame):
    import neurokit2 as nk
    df = ecg_df.copy()

    # 1) 先把时间轴和振幅整理干净
    if "time_s" not in df.columns or "value" not in df.columns:
        raise ValueError("ECG 缺少 time_s 或 value 列")

    # 排序 + 去重 + 去 NaN
    df = df.dropna(subset=["time_s", "value"]).sort_values("time_s")
    # 去掉完全相同时间戳的重复点，避免 dt 中位数被拉成 0
    dup_mask = df["time_s"].diff().fillna(0).eq(0)
    if dup_mask.any():
        df = df[~dup_mask]

    t = df["time_s"].to_numpy(float)
    sig = df["value"].to_numpy(float)
    n = len(sig)

    # 2) 采样率稳健估计：优先用列里的 fs_hz；不靠谱就用 dt 中位数反推
    fs_col = df.get("fs_hz", pd.Series([np.nan])).iloc[0]
    fs = float(fs_col) if pd.notna(fs_col) else np.nan
    if not np.isfinite(fs) or fs < 10 or fs > 1000:
        dt = np.median(np.diff(t)) if n > 2 else np.nan
        fs_est = (1.0 / dt) if (dt is not None and np.isfinite(dt) and dt > 0) else np.nan
        fs = float(fs_est) if np.isfinite(fs_est) else 130.0  # 兜底 130 Hz

    # NeuroKit 的平滑核会按 fs 算，fs 太低会出 0 核 -> 报错。强行提到 >=10。
    if fs < 10:
        print(f"[warn] ECG fs_hz={fs:.2f}Hz 太低，改为 130Hz 再做 R 峰检测")
        fs = 130.0

    # 如果信号太短，直接放弃
    if n < int(3 * fs):  # 少于约 3 秒，别逞强
        print(f"[warn] ECG 长度只有 {n} 点(<~3s)，放弃 ECG→RR")
        return None

    # 3) 试跑 ecg_process；若失败改用更稳的组合：ecg_clean + ecg_peaks(Pan-Tompkins)
    try:
        _, info = nk.ecg_process(sig, sampling_rate=fs, method="pantompkins1985")
    except Exception as e:
        print(f"[warn] neurokit.ecg_process 失败：{e}. 尝试 clean+peaks 回退。")
        sig_clean = nk.ecg_clean(sig, sampling_rate=fs, method="biosppy")
        _, info = nk.ecg_peaks(sig_clean, sampling_rate=fs, method="pantompkins1985")

    rpeaks = info.get("ECG_R_Peaks", [])
    if rpeaks is None or len(rpeaks) < 3:
        print("[warn] ECG→RR：R 峰过少，放弃")
        return None

    rr_ms = np.diff(rpeaks) / fs * 1000.0
    t_rr  = t[rpeaks[1:]]
    return pd.DataFrame({"t_s": t_rr, "rr_ms": rr_ms})

# 从 <sid>_rr.csv 读设备原生 RR
def _read_device_rr(sid: str):
    p = SRC_NORM_DIR / f"{sid}_rr.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    if not {"t_s","rr_ms"}.issubset(df.columns): return None
    return df

# 把 RR 换成 1 Hz 的心率轨，用等宽 1 秒桶求每秒均值，用于MAE，bias比较
def _hr_1hz_from_rr(rr: pd.DataFrame, t0=None, t1=None, step=1.0):
    if rr is None or rr.empty: return None
    t = rr["t_s"].to_numpy(); hr = 60000.0/rr["rr_ms"].to_numpy()
    if t0 is None: t0 = float(t.min()); 
    if t1 is None: t1 = float(t.max())
    bins = np.arange(t0, t1+step, step)
    idx = np.digitize(t, bins)-1
    out = np.full(len(bins), np.nan)
    for i in range(len(bins)):
        h = hr[idx==i]
        if len(h): out[i] = np.nanmean(h)
    return bins, out

# 简单的标注数据合法，基本规则为相邻 RR 变化幅度阈值与心率越界阈值
def _flag_rr(rr: pd.DataFrame):
    thr = float(PARAMS["rr_artifact_threshold"])
    hrmin, hrmax = PARAMS["hr_min_max_bpm"]
    rr = rr.copy()
    hr = 60000.0/rr["rr_ms"]
    flag_delta = rr["rr_ms"].pct_change().abs()>thr
    flag_delta.iloc[0]=False
    flag_hr = (hr<hrmin)|(hr>hrmax)
    rr["flagged"] = (flag_delta|flag_hr)
    return rr

# 读取 events（<sid>_events.csv），返回 DataFrame[time_s, events]
def _read_events(sid: str):
    p = SRC_NORM_DIR / f"{sid}_events.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if not {"time_s", "events"}.issubset(df.columns):
        return None
    df = df.dropna(subset=["time_s"]).sort_values("time_s")
    return df

# 读取 ACC（<sid>_acc.parquet/csv），并计算 1Hz 活动强度（0-1 归一化）
# 活动强度 = 每秒 |acc| 的均值，经最小-最大归一化。若只有 acc_mag 列则直接使用。
# 返回 (bins_sec, activity_0_1)
def _read_acc_activity_1hz(sid: str, t0=None, t1=None, step=1.0):
    """
    将三轴加速度（norm 目录下 <sid>_acc.parquet/csv）聚合为 1Hz 的活动强度曲线。

    参数
    -----
    sid : str
        被试标识（与文件名前缀一致）。函数会在 `SRC_NORM_DIR` 中查找
        `<sid>_acc.parquet` 或 `<sid>_acc.csv`。
    t0 : float | None
        计算范围的起始时间（秒）。若为 None，则使用该被试加速度数据
        的最小 `time_s` 作为起点。
    t1 : float | None
        计算范围的结束时间（秒）。若为 None，则使用该被试加速度数据
        的最大 `time_s` 作为终点。
    step : float
        聚合步长（秒）。默认 1.0，即按每秒求一次活动强度的均值。

    返回
    -----
    (bins, activity) : (np.ndarray, np.ndarray)
        `bins` 为按 `step` 等间隔划分的时间边界（秒），长度为 N；
        `activity` 为对应的 0–1 归一化活动强度（N 长度），灰色点线用于
        图上仅作运动程度参考。若文件缺失或数据不足，返回 (None, None)。

    说明
    -----
    - 活动强度的原理：若存在 `value_x/value_y/value_z` 列，则按模长
      `sqrt(x^2 + y^2 + z^2)` 得到幅值，再对每个 1 秒桶求平均；随后用
      5–95 分位数做鲁棒的最小-最大归一化，限制在 [0,1]。
    - 若未找到加速度文件或必要列，函数直接返回 (None, None)。
    """
    p = SRC_NORM_DIR / f"{sid}_acc.parquet"
    if not p.exists():
        p = SRC_NORM_DIR / f"{sid}_acc.csv"
        if not p.exists():
            return None, None
    try:
        acc = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    except Exception:
        return None, None
    if "time_s" not in acc.columns:
        return None, None

    # 计算模长或使用现成幅值
    if {"value_x", "value_y", "value_z"}.issubset(acc.columns):
        mag = np.sqrt(acc["value_x"]**2 + acc["value_y"]**2 + acc["value_z"]**2)
    else:
        return None, None

    ts = pd.to_numeric(acc["time_s"], errors="coerce")
    ok = ~(mag.isna() | ts.isna())
    mag = mag[ok].to_numpy()
    ts = ts[ok].to_numpy()
    if len(ts) < 3:
        return None, None

    if t0 is None: t0 = float(np.nanmin(ts))
    if t1 is None: t1 = float(np.nanmax(ts))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 - t0 <= 0:
        return None, None

    bins = np.arange(t0, t1 + step, step)
    idx = np.digitize(ts, bins) - 1
    act = np.full(len(bins), np.nan)
    for i in range(len(bins)):
        vals = mag[idx == i]
        if len(vals):
            act[i] = np.nanmean(np.abs(vals))

    # 0-1 归一化（鲁棒：用分位数避免尖峰支配）
    v = act[~np.isnan(act)]
    if v.size < 3:
        return bins, act  # 返回未归一化，也行
    lo, hi = np.nanpercentile(v, [5, 95])
    if hi > lo:
        act = (act - lo) / (hi - lo)
        act = np.clip(act, 0.0, 1.0)
    return bins, act

# 根据策略 PLOT_RANGE_MODE 决定绘图横轴范围
# bins: HR 曲线的时间边界（np.ndarray）
# 返回 (xmin, xmax)
def _determine_plot_range(sid: str, bins):
    if bins is None or len(bins) == 0:
        return None, None
    xmin_hr, xmax_hr = float(bins[0]), float(bins[-1])

    # 读取事件
    ev = _read_events(sid)
    has_ev = ev is not None and len(ev) >= 2 and 'time_s' in ev.columns
    if has_ev:
        ev_min = float(ev['time_s'].min())
        ev_max = float(ev['time_s'].max())
    else:
        ev_min = ev_max = None

    mode = PLOT_RANGE_MODE.lower() if isinstance(PLOT_RANGE_MODE, str) else 'auto'

    if mode == 'events' and has_ev:
        xmin, xmax = ev_min, ev_max
    elif mode == 'hr':
        xmin, xmax = xmin_hr, xmax_hr
    else:  # auto
        xmin, xmax = (ev_min, ev_max) if has_ev else (xmin_hr, xmax_hr)

    # 给一点边距，避免贴边的标签遮挡
    if xmin is not None and xmax is not None and xmax > xmin:
        pad = max(1.0, 0.01 * (xmax - xmin))
        xmin -= pad
        xmax += pad
    return xmin, xmax

# 在已有 HR 图上叠加 events（红色竖线）与 ACC 活动（灰色点线，右轴 0–1）
# bins_range: (xmin,xmax) 用于限制可视范围，可传 None
def _overlay_events_and_acc(ax, sid: str, bins_range=None):
    xmin, xmax = (None, None) if bins_range is None else bins_range

    # 叠加 ACC 活动
    acc_bins, acc_act = _read_acc_activity_1hz(sid, t0=xmin, t1=xmax)
    if acc_bins is not None and acc_act is not None:
        ax2 = ax.twinx()
        ax2.plot(acc_bins, acc_act, linestyle="-", color="0.8", alpha=0.9, linewidth=0.6, label="activity (acc)")
        ax2.set_ylabel("activity (0–1)")
        ax2.set_ylim(0, 1)
        # 避免图例重复，把第二轴的线加入主图例
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")

    # 叠加事件（竖线 + 文本标签），并打印诊断信息
    ev = _read_events(sid)
    if ev is not None and len(ev):
        ev = ev.dropna(subset=["time_s"]).sort_values("time_s").copy()
        total_events = len(ev)
        # 限定显示范围
        if xmin is not None:
            in_range = (ev["time_s"] >= xmin) & (ev["time_s"] <= (xmax if xmax is not None else ev["time_s"].max()))
            ev_in = ev[in_range]
            ev_out = ev[~in_range]
        else:
            ev_in = ev; ev_out = ev.iloc[0:0]

        # 画竖线
        for _, row in ev_in.iterrows():
            x = float(row["time_s"]) 
            ax.axvline(x=x, color="red", linestyle="-", linewidth=1.2, alpha=0.9)
                
        # 画文本标签（竖排），贴近顶端
        if len(ev_in):
            ymin, ymax = ax.get_ylim()
            ytext = ymax * 0.98
            for _, row in ev_in.iterrows():
                x = float(row["time_s"]) * 1.00001
                label = str(row.get("events", "")).strip()
                if label:
                    ax.text(x, ytext, label, color="red", fontsize=8, rotation=90,
                            va="top", ha="center", alpha=0.9, clip_on=True)
        
                # 控制台诊断：总数/可见/被裁掉
        try:
            n_in, n_out = len(ev_in), len(ev_out)
            if n_out > 0:
                dropped = ", ".join([f"{float(t):.3f}:{str(l)}" for t,l in zip(ev_out["time_s"].tolist(), ev_out.get("events", [""]*n_out))])
                print(f"[events] {sid}: total={total_events}, shown={n_in}, clipped={n_out} → {dropped}")
            else:
                print(f"[events] {sid}: total={total_events}, shown={n_in}, clipped=0")
        except Exception:
            pass

# 画单路 RR 的预览图（把逐搏 RR 映射为 1Hz HR 曲线）
def _plot_preview_single_rr(sid: str, rr: pd.DataFrame, label: str = "HR from ECG→RR"):
    if rr is None or rr.empty:
        return
    bins, hr = _hr_1hz_from_rr(rr)
    if bins is None or hr is None:
        return
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(bins, hr, label=label)
    ax.set_title(f"{sid} (preview)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("bpm")
    # ax.legend()
    ax.grid(True, alpha=.3)

    # 选择横轴范围（events 优先或 HR，取决于策略）
    xmin, xmax = _determine_plot_range(sid, bins)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
        rng = (xmin, xmax)
    else:
        rng = (bins[0], bins[-1])

    # 叠加 events 与 acc（按选定范围）
    _overlay_events_and_acc(ax, sid, bins_range=rng)
    # 图例放右上角（_overlay_里已合并第二轴图例）
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")
    
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / f"preview_{sid}.png", dpi=130)
    plt.close(fig)

def main():
    """
    选择 RR 数据的执行逻辑如下：
    - 在 2_data_norm 输出的数据中拿到分析数据。期望目录为 data/processd/norm/<settings.ACTIVE_DATA,即数据库名>
    - 对于用户指定的每个被试，分别进行设备RR 与 ECG-RR的数据获取
        - 如果只有 ECG ，则直接转换为 RR 并保存。
    - 把两路 RR 都映射成 1 Hz 心率轨，对重叠时段做 MAE、bias，算一下“被标记比例”，用于比较那个 RR 的数据质量更好
        - 把设备 RR 与 ECG-RR 的重叠部分的数据放在一起做MAE，bias。
        - 对两份 RR 进行时间对齐
        - 先看 MAE/Bias 是否在阈值内（来自 RR_COMPARE）
        - 选择 flag 标注率更低的 RR 
    - 若本数据源仅有 ECG（如 fantasia）：直接 ECG→RR，跳过建议/决策阶段；
      - 仍会按 SUBJECTS_FILTER 显示预览图：
        • 若只填 1 个短号/完整 sid，则仅绘该被试的预览；
        • 若为空或多个，仅绘第一个被试的预览。
    - 否则（同时有设备 RR 与 ECG）：生成“建议表” decision_suggested.csv 和预览图 preview_<sid>.png；
      将 decision_suggested.csv 重命名/复制为 decision.csv；可按需编辑其中的 choice_suggested 列；再次运行脚本将产出最终 RR。
    """
    # 找到所有 被测id <sid>（有 rr 或 ecg 的算）
    sids = sorted({p.stem.split("_")[0] for p in SRC_NORM_DIR.glob("*_rr.csv")}|{p.stem.split("_")[0] for p in SRC_NORM_DIR.glob("*_ecg.*")})
    # 仅挑选用户指定的被试
    all_sids = sids
    sids = sid_filter(all_sids)

    if not sids:
        print(f"[err] {SRC_NORM_DIR} 下没找到 *_rr.csv 或 *_ecg.*"); return

    # === ECG-only 快速通道（如 fantasia）：直接 ECG→RR 并落盘 ===
    has_any_device_rr = any(SRC_NORM_DIR.glob("*_rr.csv"))
    if not has_any_device_rr:
        RR_OUT_DIR.mkdir(parents=True, exist_ok=True)
        # 选择需要绘图预览的被试：
        preview_sid = None
        # 拿出选中的被试绘图
        sid_filted = sid_filter(sids)
        if sid_filted and len(sid_filted) == 1:
            # 用过滤后的列表中第一个（若存在）
            preview_sid = sid_filted[0] if len(sid_filted) > 0 else None
        else:
            preview_sid = sids[0] if len(sids) > 0 else None
        out_rows = []
        iter_sids = tqdm(sids, desc="ECG→RR", unit="sid") if tqdm else sids
        for sid in iter_sids:
            ecg = _read_ecg(sid)
            if ecg is None:
                print(f"[skip] {sid} 找不到 ECG 文件")
                continue
            rr = _ecg_to_rr(ecg)
            if rr is None or rr.empty:
                print(f"[warn] {sid} ECG→RR 失败或为空")
                continue
            out = RR_OUT_DIR / f"{sid}_rr.csv"
            rr.to_csv(out, index=False)
            out_rows.append({"subject_id": sid, "rows": len(rr), "source": "ecg_rr"})
            if preview_sid is not None and sid == preview_sid:
                _plot_preview_single_rr(sid, rr)
        # 保存一个简要概览
        if out_rows:
            pd.DataFrame(out_rows).to_csv(PREVIEW_DIR / "final_rr_summary.csv", index=False)
            print(f"[save] 最终 RR 概览 → {PREVIEW_DIR/'final_rr_summary.csv'}")
        print("[done] ECG-only 模式完成（跳过建议/决策阶段）。")
        return

    # 第一阶段：若没有决策表，就只做建议
    suggest_rows = []
    iter_sids = tqdm(sids, desc="RR compare", unit="sid") if tqdm else sids
    for sid in iter_sids:
        rr_dev = _read_device_rr(sid)
        ecg = _read_ecg(sid)
        rr_ecg = _ecg_to_rr(ecg) if ecg is not None else None
        # 打印 采样率和数据长度
        if rr_ecg is None and ecg is not None:
            fs_dbg = float(ecg.get("fs_hz", pd.Series([np.nan])).iloc[0]) if "fs_hz" in ecg.columns else float("nan")
            print(f"[dbg] {sid}: ECG rows={len(ecg)}  fs_hz_in={fs_dbg}")

        if rr_dev is None and rr_ecg is None:
            print(f"[skip] {sid} 没有 RR 也没有 ECG"); continue

        # 1Hz HR 对齐
        t0 = None
        if rr_dev is not None: t0 = rr_dev["t_s"].min()
        if rr_ecg is not None: t0 = rr_ecg["t_s"].min() if t0 is None else max(t0, rr_ecg["t_s"].min())
        t1 = None
        if rr_dev is not None: t1 = rr_dev["t_s"].max()
        if rr_ecg is not None: t1 = rr_ecg["t_s"].max() if t1 is None else min(t1, rr_ecg["t_s"].max())
        if t0 is None or t1 is None or t1-t0<10:
            print(f"[warn] {sid} 可重叠时段太短"); continue
        bins, hr_dev = _hr_1hz_from_rr(rr_dev, t0,t1) if rr_dev is not None else (None, None)
        _,    hr_ecg = _hr_1hz_from_rr(rr_ecg, t0,t1) if rr_ecg is not None else (None, None)

        # 指标
        mae=bias=np.nan; valid_dev=valid_ecg=flag_dev=flag_ecg=np.nan
        if rr_dev is not None and rr_ecg is not None:
            mask = ~np.isnan(hr_dev) & ~np.isnan(hr_ecg)
            if mask.sum()>0:
                diff = hr_dev[mask]-hr_ecg[mask]
                mae = float(np.nanmean(np.abs(diff))); bias = float(np.nanmean(diff))
        if rr_dev is not None:
            r2 = _flag_rr(rr_dev); valid_dev = 1.0 - r2["flagged"].mean(); flag_dev = r2["flagged"].mean()
        if rr_ecg is not None:
            r2 = _flag_rr(rr_ecg); valid_ecg = 1.0 - r2["flagged"].mean(); flag_ecg = r2["flagged"].mean()

        # 统计与评分（权重来自 settings.RR_SCORE；若不存在则用默认值）
        overlap_s = float(t1 - t0) if (t0 is not None and t1 is not None) else np.nan
        n_beats_device = int(len(rr_dev)) if rr_dev is not None else 0
        n_beats_ecg    = int(len(rr_ecg)) if rr_ecg is not None else 0

        # 归一化得分（越大越好）：1-flag_ratio、1-归一化MAE、1-归一化Bias
        def _nz(x): 
            return 0.0 if (x is None or (isinstance(x,float) and np.isnan(x))) else float(x)
        thr_mae  = float(RR_COMPARE.get("mae_bpm_max", 5.0))
        thr_bias = float(RR_COMPARE.get("bias_bpm_max", 3.0))
        w = RR_SCORE if 'RR_SCORE' in globals() else {"w_flag":0.5, "w_mae":0.3, "w_bias":0.2}

        def _score(flag_ratio, mae, bias):
            s_flag = max(0.0, 1.0 - _nz(flag_ratio))
            s_mae  = max(0.0, 1.0 - min(abs(_nz(mae))/thr_mae, 1.0))
            s_bias = max(0.0, 1.0 - min(abs(_nz(bias))/thr_bias, 1.0))
            return w.get("w_flag",0.5)*s_flag + w.get("w_mae",0.3)*s_mae + w.get("w_bias",0.2)*s_bias

        score_device = _score(flag_dev, mae, bias) if rr_dev is not None else -1.0
        score_ecg    = _score(flag_ecg, mae, bias) if rr_ecg is not None else -1.0

        # 建议（若两者都有：按评分；否则用仅存在的一路）
        if rr_ecg is None and rr_dev is None:
            suggest = ""
        elif rr_ecg is None:
            suggest = "device_rr"
        elif rr_dev is None:
            suggest = "ecg_rr"
        else:
            suggest = "device_rr" if score_device >= score_ecg else "ecg_rr"

        suggest_rows.append({
            "subject_id": sid,
            "has_device_rr": rr_dev is not None,
            "has_ecg": ecg is not None,
            "overlap_s": round(overlap_s, 2) if pd.notna(overlap_s) else "",
            "n_beats_device": n_beats_device if n_beats_device else "",
            "n_beats_ecg": n_beats_ecg if n_beats_ecg else "",
            "flag_ratio_device": round(flag_dev, 3) if pd.notna(flag_dev) else "",
            "flag_ratio_ecg":     round(flag_ecg, 3) if pd.notna(flag_ecg) else "",
            "valid_ratio_device": round(valid_dev, 3) if pd.notna(valid_dev) else "",
            "valid_ratio_ecg":    round(valid_ecg, 3) if pd.notna(valid_ecg) else "",
            "mae_bpm":  round(mae, 3) if pd.notna(mae) else "",
            "bias_bpm": round(bias, 3) if pd.notna(bias) else "",
            "score_device": round(score_device, 3) if score_device >= 0 else "",
            "score_ecg":    round(score_ecg, 3) if score_ecg >= 0 else "",
            "choice_suggested": suggest
        })

        # 画对比图
        fig, ax = plt.subplots(figsize=(12,3))
        if rr_dev is not None: ax.plot(bins, hr_dev, label="HR from device RR")
        if rr_ecg is not None: ax.plot(bins, hr_ecg, label="HR from ECG→RR", linestyle="--")
        tit = f"{sid}  MAE={mae:.2f}  bias={bias:+.2f} bpm"
        # ax.set_title(tit); ax.set_xlabel("time (s)"); ax.set_ylabel("bpm"); ax.legend(); ax.grid(True, alpha=.3)
        # fig.tight_layout(); fig.savefig(PREVIEW_DIR / f"preview_{sid}.png", dpi=130); plt.close(fig)
        ax.set_title(tit)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("bpm")
        ax.legend()
        ax.grid(True, alpha=.3)
         # 选择横轴范围并叠加 events/acc
        xmin, xmax = _determine_plot_range(sid, bins)
        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin, xmax)
            rng = (xmin, xmax)
        else:
            rng = (bins[0], bins[-1])
        _overlay_events_and_acc(ax, sid, bins_range=rng)
        fig.tight_layout()
        fig.savefig(PREVIEW_DIR / f"preview_{sid}.png", dpi=130)
        plt.close(fig)

    # 无决策文件则生成建议并退出
    if not DEC_FILE.exists():
        pd.DataFrame(suggest_rows).to_csv(DEC_SUGG, index=False)
        guid = (
            "【下一步怎么做】\n"
            "  1) 打开上面的建议表 decision_suggested.csv，查看每个被试两路RR的对比指标：\n"
            "     - overlap_s / n_beats_* / flag_ratio_* / valid_ratio_* / mae_bpm / bias_bpm / score_*\n"
            "  2) 如需手动修改选择，在 'choice_suggested' 列填入 device_rr 或 ecg_rr。\n"
            "  3) 将 decision_suggested.csv 重命名（或复制）为 decision.csv。\n"
            "  4) 再次运行本脚本：将按 decision.csv 的 choice_suggested 生成最终 RR 到：\n"
            f"     {RR_OUT_DIR}\n"
            "  小贴士：评分权重在 settings.RR_SCORE 中可配置（w_flag/w_mae/w_bias）；\n"
            "          阈值在 settings.RR_COMPARE（mae_bpm_max, bias_bpm_max）。"
        )
        print(f"[suggest] 已生成建议与预览，请人工确认后复制为 {DEC_FILE.name}。\n {guid} \n - 建议表：{DEC_SUGG}\n  - 预览图：{PREVIEW_DIR/'preview_<sid>.png'}")
        cheat = (
            "[guide] 如何读这张对比图 & 建议表：\n"
            f"  • 图：蓝=设备RR→HR，橙=ECG→RR→HR；标题含 MAE / bias；仅用于可视对比（逐搏RR不受影响）。\n"
            f"  • flag_ratio_* 越小越干净；valid_ratio_* 越大越好。\n"
            f"  • mae_bpm ≤ {RR_COMPARE.get('mae_bpm_max',5.0)}、|bias_bpm| ≤ {RR_COMPARE.get('bias_bpm_max',3.0)} 说明两路一致性良好。\n"
            "  • score_* 已综合伪迹率、MAE、偏差（权重见 settings.RR_SCORE）；分数更高者通常更可取。\n"
            "  • 若设备RR掉段而ECG→RR更完整，可手动改选 ecg_rr；反之亦然。"
        )
        print(cheat)
        return

    # 第二阶段：读取决策，产“最终 RR”
    dec = pd.read_csv(DEC_FILE)
    # 若用户设置过滤被试id，只定稿这些 sid；否则定稿表中所有行
    if sid_filted == 1:
        print(f"[finalize] 仅对这些被试生成最终 RR：{sorted(sid_filted)}")

    out_rows=[]
    for _, r in dec.iterrows():
        sid = str(r["subject_id"])
        choice = str(r["choice_suggested"]).strip().lower()
        rr = _read_device_rr(sid) if choice=="device_rr" else None
        if rr is None:
            ecg = _read_ecg(sid)
            rr = _ecg_to_rr(ecg) if ecg is not None else None
        if rr is None:
            print(f"[warn] {sid} 无法生成最终 RR"); continue

        RR_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = RR_OUT_DIR / f"{sid}_rr.csv"
        rr.to_csv(out, index=False)
        print(f"[2rr] {sid} → {out.name} (rows={len(rr)})")
        out_rows.append({"subject_id":sid, "rows":len(rr), "source":choice})
    if out_rows:
        pd.DataFrame(out_rows).to_csv(PREVIEW_DIR/"final_rr_summary.csv", index=False)
        print(f"[save] 最终 RR 概览 → {PREVIEW_DIR/'final_rr_summary.csv'}")
    print("[done] RR 选择完成。")
if __name__=="__main__":
    main()