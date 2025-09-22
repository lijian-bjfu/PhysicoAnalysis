# scripts/5a_windowing.py
# 功能：切窗方法和模式按 settings.windowing 定义执行切窗（cover 或 subdivide），
#       在 index.csv 中完整记录 parent_level/parent_w_id 与 level/w_id，
#       并在每个窗口文件中写入 level 字段，保证可追溯地多层细分。

from __future__ import annotations
import os, sys, re, json
from pathlib import Path
import pandas as pd
import numpy as np

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent))
        break
    _p = _p.parent
# --------------------------------------------------------
from settings import DATASETS, ACTIVE_DATA, DATA_DIR, SCHEMA, PARAMS

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# GUI 选择器
def _choose_file(title="请选择一个文件") -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        f = filedialog.askopenfilename(title=title)
        return Path(f) if f else None
    except Exception:
        return None

# 通用工具

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def fmt_s(t: float) -> str:
    if t is None or np.isnan(t):
        return "NA"
    m, s = divmod(float(t), 60.0)
    return f"{int(m):02d}:{s:06.3f}"

def read_table_if_exists(p: Path, cols_map: dict[str,str] | None = None) -> pd.DataFrame | None:
    if not p.exists(): return None
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        return None
    if cols_map:
        df = df.rename(columns=cols_map)
    return df

# 新定义 stride_s 语义的步长计算 helper
def _compute_step(win_len: float, gap: float | None) -> float:
    """
    新定义：stride_s 表示“相邻窗口之间的间隔”（gap，单位秒）。
    - gap = 0   → 窗口首尾紧贴（无重叠）。
    - gap > 0   → 相邻窗口之间留出 gap 秒空隙。
    - gap < 0   → 相邻窗口重叠 |gap| 秒。
    真实步长 step = win_len + gap。若 step <= 0 则报错。
    """
    if win_len is None:
        raise SystemExit("[error] 计算步长需要 win_len_s。")
    if gap is None:
        gap = 0.0  # 默认紧贴
    step = float(win_len) + float(gap)
    if step <= 0:
        raise SystemExit("[error] stride_s(作为间隔/重叠) 太小，导致步长 ≤ 0。请增大间隔或缩短窗长。")
    return step

# level 目录工具

def next_level_dir(base: Path) -> tuple[Path, int]:
    """
    在 base 下查找以 'level' 开头的文件夹，解析最大编号，返回 (下一个 levelN 路径, N)。
    若不存在任何 level* 目录，则返回 (base/level1, 1)。
    """
    max_n = 0
    if base.exists():
        for d in base.iterdir():
            if d.is_dir():
                m = re.match(r"^level(\d+)$", d.name.strip())
                if m:
                    try:
                        n = int(m.group(1))
                        if n > max_n:
                            max_n = n
                    except ValueError:
                        pass
    return base / f"level{max_n + 1}", (max_n + 1)

def parse_level_from_dir(d: Path) -> int:
    m = re.match(r"^level(\d+)$", d.name)
    return int(m.group(1)) if m else 1

# 配置解析

def _resolve_cfg():
    ds = DATASETS[ACTIVE_DATA]
    wcfg = ds["windowing"]
    paths = ds["paths"]

    SRC_RR_DIR    = (DATA_DIR / paths["confirmed"]).resolve()
    SRC_OTHER_DIR = (DATA_DIR / paths["norm"]).resolve()
    OUT_BASE      = (DATA_DIR / paths["windowing"]).resolve()

    method  = wcfg.get("method", "cover").lower().strip()  # cover | subdivide
    preview_sids = ds.get("preview_sids", [])
    if not isinstance(preview_sids, (list, tuple)):
        preview_sids = []

    use_mode = wcfg.get("use", "events").strip()
    apply_to = wcfg.get("apply_to", ["rr"])  # e.g. ["rr","resp","acc","events"]

    return {
        "paths": {"rr": SRC_RR_DIR, "other": SRC_OTHER_DIR, "out": OUT_BASE},
        "wcfg": wcfg,
        "use_mode": use_mode,
        "apply_to": apply_to,
        "method": method,
        "preview_sids": preview_sids
    }

# 当用户将结尾的值设为特别大的时候，比如999999专这样的数，直接取结尾。
def _rr_bounds(sid: str, cfg: dict) -> tuple[float, float]:
    rr = read_table_if_exists(cfg["paths"]["rr"]/f"{sid}_rr.csv", {"t_s":"t_s","rr_ms":"rr_ms"})
    if rr is None:
        rr = read_table_if_exists(cfg["paths"]["rr"]/f"{sid}_rr.parquet", {"t_s":"t_s","rr_ms":"rr_ms"})
    if rr is None or rr.empty:
        raise SystemExit(f"[error] {sid}: 找不到 RR 数据，无法切窗。")
    return float(rr["t_s"].min()), float(rr["t_s"].max())

# 被试列表（以 confirmed 下 *_rr.* 为准）

def list_subjects(src_rr_dir: Path) -> list[str]:
    return sorted({p.stem.split("_")[0] for p in src_rr_dir.glob("*_rr.*")})

# 事件读取（prefer_dir 可覆盖事件查找路径，用于按父层同目录读取）

def load_events_for_sid(cfg: dict, sid: str, prefer_dir: Path | None = None) -> pd.DataFrame | None:
    if prefer_dir is not None:
        cand = prefer_dir / f"{sid}_events.csv"
        df = read_table_if_exists(cand)
        if df is not None:
            return df.rename(columns={SCHEMA["events"]["t"]:"time_s", SCHEMA["events"]["label"]:"events"}) \
                     .dropna(subset=["time_s","events"]).sort_values("time_s")
    cand = cfg["paths"]["other"] / f"{sid}_events.csv"
    df = read_table_if_exists(cand)
    if df is None:
        cand = cfg["paths"]["other"] / f"{sid}_events.parquet"
        df = read_table_if_exists(cand)
    if df is None: return None
    return df.rename(columns={SCHEMA["events"]["t"]:"time_s", SCHEMA["events"]["label"]:"events"}) \
             .dropna(subset=["time_s","events"]).sort_values("time_s")

# 预览图

def draw_preview(sid: str, cfg: dict, windows: list[dict], out_dir: Path):
    import matplotlib.pyplot as plt
    rr = read_table_if_exists(cfg["paths"]["rr"]/f"{sid}_rr.csv", {"t_s":"t_s","rr_ms":"rr_ms"})
    if rr is None:
        rr = read_table_if_exists(cfg["paths"]["rr"]/f"{sid}_rr.parquet", {"t_s":"t_s","rr_ms":"rr_ms"})
    hr = read_table_if_exists(cfg["paths"]["other"]/f"{sid}_hr.csv", {SCHEMA["hr"]["t"]:"t_s", SCHEMA["hr"]["v"]:"bpm"})
    if hr is None:
        hr = read_table_if_exists(cfg["paths"]["other"]/f"{sid}_hr.parquet", {SCHEMA["hr"]["t"]:"t_s", SCHEMA["hr"]["v"]:"bpm"})
    resp = read_table_if_exists(cfg["paths"]["other"]/f"{sid}_resp.csv", {SCHEMA["resp"]["t"]:"time_s", SCHEMA["resp"]["v"]:"value"})
    if resp is None:
        resp = read_table_if_exists(cfg["paths"]["other"]/f"{sid}_resp.parquet", {SCHEMA["resp"]["t"]:"time_s", SCHEMA["resp"]["v"]:"value"})
    acc = read_table_if_exists(cfg["paths"]["other"]/f"{sid}_acc.csv", {SCHEMA["acc"]["t"]:"time_s", SCHEMA["acc"]["vx"]:"value_x", SCHEMA["acc"]["vy"]:"value_y", SCHEMA["acc"]["vz"]:"value_z"})
    if acc is None:
        acc = read_table_if_exists(cfg["paths"]["other"]/f"{sid}_acc.parquet", {SCHEMA["acc"]["t"]:"time_s", SCHEMA["acc"]["vx"]:"value_x", SCHEMA["acc"]["vy"]:"value_y", SCHEMA["acc"]["vz"]:"value_z"})
    eve = load_events_for_sid(cfg, sid)

    import matplotlib
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 4))
    if rr is not None and len(rr):
        hr1 = 60000.0/rr["rr_ms"].clip(lower=1)
        ax.plot(rr["t_s"], hr1, label="HR from RR", linewidth=1.2)
    if hr is not None and len(hr):
        ax.plot(hr["t_s"], hr["bpm"], label="Device HR", linewidth=1.0, alpha=0.8)
    if resp is not None and len(resp):
        ax2 = ax.twinx(); ax2.plot(resp["time_s"], resp["value"], label="Resp", linewidth=0.8, alpha=0.6)
        ax2.set_ylabel("Resp")
    if acc is not None and len(acc):
        g = np.sqrt(acc["value_x"]**2 + acc["value_y"]**2 + acc["value_z"]**2)
        ax.plot(acc["time_s"], g, label="ACC|g|", linewidth=0.6, alpha=0.35, linestyle="--", color="gray")
    if eve is not None and len(eve):
        for _, r in eve.iterrows():
            ax.axvline(float(r["time_s"]), color="red", alpha=0.7, linewidth=0.8)
    for w in windows:
        ax.axvspan(w["t_start_s"], w["t_end_s"], color="khaki", alpha=0.25)
    ax.set_title(f"{sid} 窗口预览")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.legend(loc="upper right", fontsize=8)
    out_png = out_dir / f"preview_{sid}.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[plot] 预览 → {out_png}")

# 定义窗口（不落盘，仅输出时间对 + 含义）

def build_windows_cover_for_subject(sid: str, cfg: dict) -> list[dict]:
    wcfg = cfg["wcfg"]; mode = cfg["use_mode"];
    label_tpl = wcfg.get("labeling", {}).get(mode, "{s:.1f}-{e:.1f}")
    windows: list[dict] = []

    # 统一最小时长（秒），由 settings.PARAMS.min_window 配置，默认 1.0s
    min_win = PARAMS["min_window"]
    def add_win(s: float, e: float, meaning: str):
        if s is None or e is None:
            return
        if e <= s:
            print(f"[warn] {sid}: 反序/零长度窗口被拦截 s={s}, e={e}, meaning={meaning}")
            return
        if (e - s) < min_win:
            print(f"[warn] {sid}: 窗口小于最小时长 {min_win}s 被拦截 s={s}, e={e}, meaning={meaning}")
            return
        windows.append({
            "subject_id": sid,
            "t_start_s": float(s),
            "t_end_s": float(e),
            "meaning": meaning,
            "parent_level": 0,
            "parent_w_id": 0,
        })

    def load_events() -> pd.DataFrame | None:
        m = wcfg["modes"].get("events", {})
        events_dir = (DATA_DIR / m.get("events_path", "")).resolve()
        # 统一走全局 helper，确保与预览读取一致。
        return load_events_for_sid(cfg, sid, prefer_dir=events_dir)

    if mode == "events":
        pairs = wcfg["modes"]["events"].get("pairs", [])
        ev = load_events()
        if ev is None:
            raise SystemExit(f"[error] {sid}：找不到事件文件用于 events 模式。")
        ev_map = {str(r["events"]).strip(): float(r["time_s"]) for _, r in ev.iterrows()}
        for e0, e1 in pairs:
            if e0 in ev_map and e1 in ev_map:
                s, e = ev_map[e0], ev_map[e1]
                add_win(s, e, label_tpl.format(e0=e0, e1=e1, s=s, e=e))

    elif mode == "single":
        m = wcfg["modes"]["single"]
        s = m.get("start_s"); e = m.get("end_s"); w = m.get("win_len_s"); a = m.get("anchor_time_s")
        # 结尾设定时间锁定到实际数据时间，min(exp, real)
        tmin, tmax = _rr_bounds(sid, cfg)
        if s is not None and e is not None:
            s1, e1 = max(float(s), tmin), min(float(e), tmax)
            add_win(s1, e1, label_tpl.format(s=s1, e=e1))
        elif s is not None and w is not None:
            s1, e1 = max(float(s), tmin), min(float(s)+float(w), tmax)
            add_win(s1, e1, label_tpl.format(s=s1, e=e1))
        elif a is not None and w is not None:
            s1, e1 = max(float(s), tmin), min(float(s)+float(w), tmax)
            add_win(s1, e1, label_tpl.format(s=s1, e=e1))
        else:
            raise SystemExit("[error] single 模式参数不足（需 [start,end] 或 [start,win] 或 [anchor,win]）")

    elif mode in ("sliding","slid"):
        m = wcfg["modes"]["sliding"] if mode=="sliding" else wcfg["modes"]["slid"]
        win_len = m.get("win_len_s") or m.get("win_len")
        gap     = m.get("stride_s", 0.0)
        if win_len is None:
            raise SystemExit("[error] sliding 模式缺少 win_len_s")

        # 用户显式给出的起止（允许缺省）；缺省则用 RR 的真实边界
        user_s0 = m.get("start_s", None)
        user_e0 = m.get("end_s", None)

        # 真实 RR 边界
        tmin, tmax = _rr_bounds(sid, cfg)

        # 起点：若用户未给，则用 tmin；若给了则与 tmin 取较大值（不早于数据起点）
        s0 = tmin if user_s0 is None else max(float(user_s0), tmin)

        # 终点：若用户未给，则用 tmax；若给了则与 tmax 取较小值（不晚于数据终点）
        e0 = tmax if user_e0 is None else min(float(user_e0), tmax)

        if e0 - s0 < float(win_len):
            print(f"[warn] {sid}: 有效区间不足一个窗（{fmt_s(s0)}~{fmt_s(e0)}，win={win_len}s），本被试无窗口。")
        else:
            step = _compute_step(float(win_len), gap)
            approx_n = int(max(0, (float(e0) - float(s0) - float(win_len)) / step))
            if step < 0.01 * float(win_len) or approx_n > 10000:
                print(f"[warn] {sid}: 步长过小或窗口数过多 step={step:.6f}, win={float(win_len):.6f}, 预计窗口数≈{approx_n}")         
            t = float(s0)
            while t + float(win_len) <= float(e0) + 1e-9:
                s_cur, e_cur = t, t + float(win_len)
                add_win(s_cur, e_cur, label_tpl.format(s=s_cur, e=e_cur))
                t += step

    elif mode == "events_single":
        m = wcfg["modes"]["events_single"]
        ev = load_events()
        #拿到 events.csv
        events_dir = (DATA_DIR / wcfg["modes"].get("events", {}).get("events_path", "")).resolve()
        if ev is None:
            raise SystemExit(f"[error] events_single 模式需要事件表，但在 {events_dir} 未找到 {sid}_events.csv 或 .parquet")
        
        # 来自事件列表，去settings 填写。不可以为 None
        anchor = m.get("anchor_event", None)
        off = float(m.get("offset_s", 0.0) or 0.0)
        w = m.get("win_len_s", None)

        if not anchor or w is None: 
            raise SystemExit("[error] events_single 缺少 anchor_event 或 win_len_s")

        t_anchor = float(ev.loc[ev["events"]==anchor, "time_s"].min())
        add_win(t_anchor + off, t_anchor + off + w, label_tpl.format(anchor=anchor, off=off, w=w))

    elif mode == "events_sliding":
        m = wcfg["modes"]["events_sliding"]
        ev = load_events()
        if ev is None:
            raise SystemExit("[error] events_sliding 需要事件文件")
        seg = m.get("segment", None)
        w = m.get("win_len_s")
        gap = m.get("stride_s", 0.0)
        if not seg or w is None:
            raise SystemExit("[error] events_sliding 需要 segment, win_len_s")

        # 事件时间
        ev_map = {str(r["events"]).strip(): float(r["time_s"]) for _, r in ev.iterrows()}
        if seg[0] not in ev_map or seg[1] not in ev_map:
            raise SystemExit(f"[error] events_sliding 缺少事件 {seg}")

        # 先用事件确定，再与 RR 边界钳位
        s0_evt, e0_evt = ev_map[seg[0]], ev_map[seg[1]]
        tmin, tmax = _rr_bounds(sid, cfg)
        s0 = max(float(s0_evt), tmin)
        e0 = min(float(e0_evt), tmax)

        if e0 - s0 < float(w):
            print(f"[warn] {sid}: 事件区间不足一个窗（{fmt_s(s0)}~{fmt_s(e0)}，win={w}s），本被试无窗口。")
        else:
            step = _compute_step(float(w), gap)
            approx_n = int(max(0, (float(e0) - float(s0) - float(w)) / step))
            if step < 0.01 * float(w) or approx_n > 10000:
                print(f"[warn] {sid}: 步长过小或窗口数过多 step={step:.6f}, win={float(w):.6f}, 预计窗口数≈{approx_n}")
            t = float(s0)
            while t + float(w) <= float(e0) + 1e-9:
                s_cur, e_cur = t, t + float(w)
                add_win(s_cur, e_cur, label_tpl.format(s=s_cur, e=e_cur))
                t += step

    elif mode == "single_sliding":
        m = wcfg["modes"]["single_sliding"]
        w = m.get("win_len_s")
        if w is None:
            raise SystemExit("[error] single_sliding 需提供 win_len_s")
        gap = m.get("stride_s", 0.0)

        # 用户可选提供 start_s/end_s；缺省则用 RR 边界
        user_s0 = m.get("start_s", None)
        user_e0 = m.get("end_s", None)
        tmin, tmax = _rr_bounds(sid, cfg)
        s0 = tmin if user_s0 is None else max(float(user_s0), tmin)
        e0 = tmax if user_e0 is None else min(float(user_e0), tmax)

        if e0 - s0 < float(w):
            print(f"[warn] {sid}: 可用范围不足一个窗（{fmt_s(s0)}~{fmt_s(e0)}，win={w}s），本被试无窗口。")
        else:
            step = _compute_step(float(w), gap)
            t = float(s0)
            while t + float(w) <= float(e0) + 1e-9:
                add_win(t, t+float(w), label_tpl.format(s=s0, e=e0, w=w, step=step))
                t += step
    else:
        raise SystemExit(f"[error] 未知模式：{mode}")

    return windows

# subdivide：从父层 index.csv 的相同 w_id 为每个被试推断父窗区间

def build_windows_subdivide(cfg: dict) -> tuple[list[dict], Path, int, int]:
    parent_example = _choose_file("请选择上一层的某个窗口文件（如 <sid>_rr_w03.csv）")
    if not parent_example or not parent_example.exists():
        raise SystemExit("[error] 你没有选择有效的窗口文件，无法细分。")
    parent_dir = parent_example.parent
    parent_level = parse_level_from_dir(parent_dir)
    fname = parent_example.name
    try:
        # <sid>_<signal>_wNN.csv → 解析父窗 wNN 与 sid（仅用于展示，不强依赖）
        wtoken = fname.rsplit("_", 1)[-1].split(".")[0]  # wNN
        parent_w_id = int(wtoken.replace("w", ""))
    except Exception:
        raise SystemExit(f"[error] 无法从文件名解析父窗口编号：{fname}")

    idx_path = parent_dir / "index.csv"
    if not idx_path.exists():
        raise SystemExit(f"[error] 未在 {parent_dir} 找到 index.csv，无法细分。")
    idx_df = pd.read_csv(idx_path)
    # 校验所需列
    need_cols = {"subject_id","w_id","t_start_s","t_end_s"}
    if not need_cols.issubset(set(idx_df.columns)):
        raise SystemExit("[error] 父层 index.csv 缺少必要列：subject_id,w_id,t_start_s,t_end_s。")

    # 对每个被试，寻找父窗 parent_w_id 的时段
    sids = list_subjects(cfg["paths"]["rr"])
    out_windows: list[dict] = []
    for sid in sids:
        row = idx_df[(idx_df["subject_id"]==sid) & (idx_df["w_id"]==parent_w_id)]
        if row.empty:
            # 若该被试没有这个父窗，跳过
            continue
        s0 = float(row.iloc[0]["t_start_s"]); e0 = float(row.iloc[0]["t_end_s"])
        # 依据当前模式在 [s0,e0] 内生成子窗
        mode = cfg["use_mode"]; wcfg = cfg["wcfg"]; 
        label_tpl = wcfg.get("labeling", {}).get(mode, "{s:.1f}-{e:.1f}")
        def add_child(s: float, e: float, meaning: str):
            if s is None or e is None: return
            if e <= s: return
            out_windows.append({
                "subject_id": sid,
                "t_start_s": float(s),
                "t_end_s": float(e),
                "meaning": meaning,
                "parent_level": int(parent_level),
                "parent_w_id": int(parent_w_id),
            })
        if mode in ("sliding","slid"):
            m = wcfg["modes"]["sliding"] if mode=="sliding" else wcfg["modes"]["slid"]
            w   = m.get("win_len_s") or m.get("win_len")
            gap = m.get("stride_s")
            st  = _compute_step(w, gap)
            t = s0
            while t + w <= e0 + 1e-9:
                add_child(t, t+w, f"parent[w{parent_w_id:02d}] -> {fmt_s(t)}+{w:.0f}s")
                t += st
        elif mode == "single":
            m  = wcfg["modes"]["single"]
            w  = m.get("win_len_s")
            anchor = (s0 + e0)/2.0 if m.get("anchor_time_s") is None else float(m.get("anchor_time_s"))
            s = max(s0, anchor)
            e = min(e0, s + w) if w else e0
            min_win = PARAMS["min_window"]
            if (e - s) >= min_win:
                add_child(s, e, f"parent[w{parent_w_id:02d}] -> single[{fmt_s(s)},{fmt_s(e)}]")
            else:
                print(f"[warn] {sid}: 细分single窗口小于最小时长 {min_win}s 被拦截 s={s}, e={e}")
        else:
            # 细分不直接允许 events 再切（避免跨层事件偏差）
            pass
    return out_windows, parent_dir, parent_level, parent_w_id

# 将各信号在窗口内切出并保存

def _save_windows_for_subject(sid: str, cfg: dict, windows_df: pd.DataFrame, out_dir: Path, level: int) -> dict:
    """
    保存 <sid>_<signal>_wNN.csv，并在文件内写入 w_id, level, meaning 列。
    返回每个 signal 实际保存窗口数。
    """
    def _load_signal(signal: str) -> tuple[pd.DataFrame | None, str]:
        # cover：从 confirmed/norm 读取
        def _try_load(paths: list[Path], cols_map: dict[str, str], tcol_name: str) -> tuple[pd.DataFrame | None, str]:
            for p in paths:
                df = read_table_if_exists(p, cols_map)
                if df is not None and len(df):
                    return df, tcol_name
            return None, ""
        if signal == "rr":
            paths = [cfg["paths"]["rr"] / f"{sid}_rr.csv", cfg["paths"]["rr"] / f"{sid}_rr.parquet"]
            return _try_load(paths, {"t_s":"t_s", "rr_ms":"rr_ms"}, "t_s")
        if signal == "resp":
            paths = [cfg["paths"]["other"] / f"{sid}_resp.csv", cfg["paths"]["other"] / f"{sid}_resp.parquet"]
            return _try_load(paths, {SCHEMA["resp"]["t"]:"time_s", SCHEMA["resp"]["v"]:"value"}, "time_s")
        if signal == "acc":
            paths = [cfg["paths"]["other"] / f"{sid}_acc.csv", cfg["paths"]["other"] / f"{sid}_acc.parquet"]
            return _try_load(paths, {SCHEMA["acc"]["t"]:"time_s", SCHEMA["acc"]["vx"]:"value_x", SCHEMA["acc"]["vy"]:"value_y", SCHEMA["acc"]["vz"]:"value_z"}, "time_s")
        if signal == "events":
            paths = [cfg["paths"]["other"] / f"{sid}_events.csv", cfg["paths"]["other"] / f"{sid}_events.parquet"]
            return _try_load(paths, {SCHEMA["events"]["t"]:"time_s", SCHEMA["events"]["label"]:"events"}, "time_s")
        return None, ""

    saved_counts = {}
    for signal in cfg["apply_to"]:
        df_sig, tcol = _load_signal(signal)
        if df_sig is None or len(df_sig)==0:
            print(f"[skip] {sid} {signal}: 未找到或为空，跳过。")
            continue
        n_saved = 0
        missing_ids = []
        for r in windows_df.itertuples(index=False):
            s, e = float(r.t_start_s), float(r.t_end_s)
            w_id = int(r.w_id)
            sub = df_sig[(df_sig[tcol] >= s) & (df_sig[tcol] < e)].copy()
            if sub.empty:
                print(f"[warn] {sid} {signal}: w{w_id:02d} 窗口无数据 [{s}, {e})，可能因数据缺失或时间不覆盖")
                missing_ids.append(w_id)
                continue
            sub["w_id"] = w_id
            sub["level"] = int(level)
            if signal != "events":  # events 原样保留，不强行加 meaning 列
                sub["meaning"] = r.meaning
            out_path = out_dir / f"{sid}_{signal}_w{w_id:02d}.csv"
            sub.to_csv(out_path, index=False)
            n_saved += 1
        expect = len(windows_df)
        if n_saved < expect:
            miss_str = ",".join([f"w{int(x):02d}" for x in missing_ids]) if missing_ids else ""
            print(f"[warn] {sid} {signal}: 实际保存 {n_saved}/{expect}，缺失窗口：{miss_str}")
        saved_counts[signal] = n_saved
        print(f"[save] {sid} {signal}: {n_saved} 个窗口 → {out_dir}")
    return saved_counts

# 主流程

def main():
    cfg = _resolve_cfg()
    paths = cfg["paths"]

    # 预览目标：仅绘制 DATASETS[ACTIVE_DATA]['preview_sids'][0]；未配置则不预览
    wcfg = cfg["wcfg"]
    prev_list = cfg.get("preview_sids", [])
    preview_target = prev_list[0] if isinstance(prev_list, (list, tuple)) and len(prev_list) > 0 else None

    # 输出目录与当前层级 level
    if cfg["method"] == "subdivide":
        OUT_DIR, level = next_level_dir(paths["out"])  # 自动 +1 层
        ensure_dir(OUT_DIR)
    else:
        out_level = os.getenv("WINDOWING_LEVEL", "level1")
        OUT_DIR = ensure_dir(paths["out"] / out_level)
        level = parse_level_from_dir(OUT_DIR)

    idx_path = OUT_DIR / "index.csv"
    log_path = OUT_DIR / "log.txt"

    print(f"[windowing] dataset={ACTIVE_DATA}  method={cfg['method']}  mode={cfg['use_mode']}")
    print(f"[paths] RR={paths['rr']}  OTHER={paths['other']}  OUT={OUT_DIR}")
    print(f"[level] 输出目录为：{OUT_DIR} (level={level})")
    print(f"[apply_to] {cfg['apply_to']}")
    if cfg["use_mode"] in ("sliding","slid","events_sliding","single_sliding"):
        print("[note] 新语义：stride_s 表示相邻窗口之间的“间隔”（秒）。负值表示重叠，0 表示紧贴，正值表示留白。")
        print("[note] 保存阶段采用半开区间 [s,e)，用于避免相邻窗口边界的双重计数。")

    sids = list_subjects(paths["rr"])
    if not sids:
        raise SystemExit("[error] 在 confirmed/ 未发现 *_rr.*")

    # 生成窗口列表（cover：按每被试模式生成；subdivide：按父层 index 推断每被试父窗，再按模式细分）
    all_windows = []
    parent_level = 0; parent_w_id = 0; parent_dir = None

    # Add preview_done flag to ensure only one preview is generated
    preview_done = False

    if cfg["method"] == "cover":
        for sid in tqdm(sids, desc="切窗"):
            wins = build_windows_cover_for_subject(sid, cfg)
            # 增加层级信息（父层为 0/0）
            for w in wins:
                w["level"] = level
                w["parent_level"] = 0
                w["parent_w_id"] = 0
            if wins:
                times = ", ".join([f"[{fmt_s(w['t_start_s'])} ~ {fmt_s(w['t_end_s'])}]" for w in wins])
                print(f"[{sid}] windows: {len(wins)} → {times}")
                if (not preview_done) and (preview_target is not None) and (sid == preview_target):
                    draw_preview(sid, cfg, wins, OUT_DIR)
                    preview_done = True
            all_windows.extend(wins)
    else:
        # subdivide：一次性为所有被试细分相同 parent_w_id
        wins, parent_dir, parent_level, parent_w_id = build_windows_subdivide(cfg)
        if not wins:
            raise SystemExit("[warn] 没有生成任何窗口（可能所选父窗在部分被试中不存在）。")
        # 标注本层级
        for w in wins:
            w["level"] = level
        # 打印摘要 + 可选预览
        for sid in sorted(set([w["subject_id"] for w in wins])):
            s = [w for w in wins if w["subject_id"]==sid]
            times = ", ".join([f"[{fmt_s(x['t_start_s'])} ~ {fmt_s(x['t_end_s'])}]" for x in s])
            print(f"[{sid}] windows (subdivide from L{parent_level} w{parent_w_id:02d}): {len(s)} → {times}")
            if s and (not preview_done) and (preview_target is not None) and (sid == preview_target):
                draw_preview(sid, cfg, s, OUT_DIR)
                preview_done = True
        all_windows = wins

    if not all_windows:
        raise SystemExit("[warn] 没有生成任何窗口，请检查 settings.windowing 的方法与参数。")

    # 按被试排序并赋予 w_id（每被试从 1 递增），并生成 lineage_path
    df = pd.DataFrame(all_windows).sort_values(["subject_id","t_start_s"]).reset_index(drop=True)
    df["w_id"] = df.groupby("subject_id").cumcount() + 1
    df["lineage_path"] = df.apply(
        lambda r: (f"L{int(r['parent_level'])}:w{int(r['parent_w_id']):02d} -> " if int(r['parent_level'])>0 else "root -> ") +
                  f"L{int(r['level'])}:w{int(r['w_id']):02d}", axis=1)

    # 写 index.csv（含层级追溯）
    cols = ["subject_id","level","w_id","parent_level","parent_w_id","t_start_s","t_end_s","meaning","lineage_path"]
    df[cols].to_csv(idx_path, index=False)
    print(f"[save] index.csv → {idx_path} (rows={len(df)})")

    # 写 log.txt（中文，人类可读；仅展示一个代表被试的详细列表）
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("【切窗日志】\n")
        f.write(f"dataset: {ACTIVE_DATA}\n")
        f.write(f"method: {cfg['method']}  mode: {cfg['use_mode']}\n")
        if cfg['method'] == 'subdivide':
            f.write(f"父层目录: {parent_dir}\n父层级: L{parent_level}  父窗口: w{parent_w_id:02d}\n")
        f.write(f"输入路径：RR={paths['rr']}\n")
        f.write(f"辅助路径：OTHER={paths['other']}\n")
        f.write(f"输出路径：{OUT_DIR} (level={level})\n")
        f.write(f"被试数量：{len(df['subject_id'].unique())}\n")
        f.write(f"窗口总数：{len(df)}\n\n")
        f.write("每窗含义（meaning）字段按 settings.windowing.labeling[mode] 模板生成；时间格式为 mm:ss.mmm。\n\n")
        # 仅挑第一个被试做详细列举
        sid0 = df['subject_id'].iloc[0]
        sub0 = df[df['subject_id']==sid0]
        f.write(f"示例被试：{sid0}（{len(sub0)} 个窗）\n")
        for _, r in sub0.iterrows():
            dur = float(r['t_end_s']) - float(r['t_start_s'])
            f.write(
                f"  · L{int(r['level'])}:w{int(r['w_id']):02d}  [{fmt_s(r['t_start_s'])} ~ {fmt_s(r['t_end_s'])}]  时长={fmt_s(dur)}  parent=L{int(r['parent_level'])}:w{int(r['parent_w_id']):02d}  {r['meaning']}\n"
            )
    print(f"[save] log.txt → {log_path}")

    # 真正落盘：对每个被试保存 RR/RESP/ACC/EVENTS 的窗口数据（含 level 与 w_id）
    all_saved = []
    for sid, sub in df.groupby("subject_id"):
        counts = _save_windows_for_subject(sid, cfg, sub[["t_start_s","t_end_s","meaning","w_id"]], OUT_DIR, level)
        all_saved.append((sid, counts))

    for sid, c in all_saved:
        detail = ", ".join([f"{k}:{v}" for k,v in c.items()])
        print(f"[done] {sid} → {detail}")

if __name__ == "__main__":
    main()