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
    apply_to = wcfg.get("apply_to", ["rr"])  # e.g. ["rr","resp","acc","events","hr"]

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

    # 统一最小时长（秒），由 settings.PARAMS.min_window 配置
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
        # 相邻事件成窗
        ev = load_events()
        if ev is None or ev.empty:
            evpath = wcfg["modes"].get("events", {})
            raise SystemExit(
                f"[error] {sid}：找不到事件文件用于 events 模式。\n  "
                f"请在 {evpath} 指定的文件夹下检查是否存在 {sid}_events.csv/.parquet，或使用其他模式切窗"
            )
        ev = ev.sort_values("time_s").reset_index(drop=True)
        if len(ev) < 2:
            print(f"[warn] {sid}: 事件数少于 2 个，无法基于相邻事件切窗。")
        else:
            # 逐对相邻事件生成窗口 [event[i], event[i+1]]
            for i in range(len(ev) - 1):
                s = float(ev.loc[i, "time_s"])
                e = float(ev.loc[i+1, "time_s"])
                e0 = str(ev.loc[i, "events"]).strip()
                e1 = str(ev.loc[i+1, "events"]).strip()
                add_win(s, e, label_tpl.format(e0=e0, e1=e1, s=s, e=e))

    elif mode == "single":
        # 新语义：single 的 start_s / end_s 为**相对数据起点 tmin 的偏移秒数**。
        # 仅支持两种用法： [start_s, end_s]  或  [start_s, win_len_s]（二选一）。
        # 不再使用 anchor_time_s。
        m = wcfg["modes"]["single"]
        off_start = m.get("start_s", None)
        off_end   = m.get("end_s", None)
        win_len   = m.get("win_len_s", None)

        if off_start is None:
            raise SystemExit("[error] single 模式需提供 start_s（相对数据起点的偏移秒数）")

        # 真实 RR 边界
        tmin, tmax = _rr_bounds(sid, cfg)

        # 绝对起点：起点 = tmin + start_s，并与 tmin 夹紧
        s_abs = max(float(tmin) + float(off_start), float(tmin))

        # 二选一：end_s（相对 tmin 的偏移）或 win_len_s
        if (off_end is not None) and (win_len is not None):
            raise SystemExit("[error] single 模式参数冲突：end_s 与 win_len_s 只能二选一。")

        if off_end is not None:
            # 绝对终点 = tmin + end_s，再与 tmax 夹紧
            e_abs = min(float(tmin) + float(off_end), float(tmax))
        elif win_len is not None:
            e_abs = min(s_abs + float(win_len), float(tmax))
        else:
            raise SystemExit("[error] single 模式参数不足，end_s 或 win_len_s 至少要设定一个")

        # 合法性检查
        if e_abs <= s_abs:
            raise SystemExit(f"[error] single 窗口反序/零长度：start={fmt_s(s_abs)} ≥ end={fmt_s(e_abs)}")
        if (e_abs - s_abs) < float(PARAMS["min_window"]):
            raise SystemExit(f"[error] single 窗口短于最小时长 {PARAMS['min_window']}s："
                             f"start={fmt_s(s_abs)} end={fmt_s(e_abs)}")

        # 记录窗口
        add_win(
            s_abs,
            e_abs,
            label_tpl.format(
                s=s_abs, e=e_abs,
                start=float(off_start),
                end=(None if off_end is None else float(off_end)),
                w=(None if win_len is None else float(win_len))
            )
        )


    elif mode == "sliding":
        m = wcfg["modes"]["sliding"]
        w = m.get("win_len_s")
        if w is None or float(w) <= 0:
            raise SystemExit("[error] sliding 模式必须设定窗口长度且必须大于 0，请在 settings.windowing.modes.sliding.win_len_s 中设置。")

        # 新语义：
        # - start_s / end_s 解释为相对数据起点 tmin 的偏移秒数；
        # - 若为 None，则分别使用数据起点/终点，并打印警告；
        # - stride_s 表示“相邻窗口之间的间隔（gap）”，可为正/负/零；None 视为 0（紧贴）。
        gap = m.get("stride_s", None)
        gap = 0.0 if gap is None else float(gap)

        # 真实 RR 边界
        tmin, tmax = _rr_bounds(sid, cfg)

        off_start = m.get("start_s", None)
        off_end   = m.get("end_s", None)

        if off_start is None:
            s_abs = float(tmin)
            print(f"[warn] {sid}: sliding.start_s 未设置，使用数据起点 {fmt_s(tmin)} 作为开始。")
        else:
            s_abs = max(float(tmin) + float(off_start), float(tmin))

        if off_end is None:
            e_abs = float(tmax)
            print(f"[warn] {sid}: sliding.end_s 未设置，使用数据终点 {fmt_s(tmax)} 作为结束。")
        else:
            e_abs = min(float(tmin) + float(off_end), float(tmax))

        if e_abs - s_abs < float(w):
            print(f"[warn] {sid}: 有效区间不足一个窗（{fmt_s(s_abs)}~{fmt_s(e_abs)}，win={float(w):.0f}s），本被试无窗口。")
        else:
            step = _compute_step(float(w), gap)  # step = win_len + gap；gap<0 表示重叠
            # 仅提示：若步长极小导致窗口数可能过多
            approx_n = int(max(0, (float(e_abs) - float(s_abs) - float(w)) / step)) if step > 0 else 0
            if step < 0.01 * float(w) or approx_n > 10000:
                print(f"[warn] {sid}: 步长过小或窗口数过多 step={step:.6f}, win={float(w):.6f}, 预计窗口数≈{approx_n}")

            t = float(s_abs)
            while t + float(w) <= float(e_abs) + 1e-9:
                s_cur, e_cur = t, t + float(w)
                add_win(s_cur, e_cur, label_tpl.format(s=s_cur, e=e_cur, w=float(w), step=step))
                t += step

    elif mode == "events_offset":
        # 基于相邻事件成窗，并按 offset[窗号]=缩进秒数，对称内缩两端
        m = wcfg["modes"].get("events_offset", {})
        # 事件文件路径优先使用本模式的 events_path
        events_dir = (DATA_DIR / m.get("events_path", "")).resolve()
        ev = load_events_for_sid(cfg, sid, prefer_dir=events_dir)
        if ev is None or ev.empty:
            raise SystemExit(f"[error] {sid}：events_offset 模式需要事件表，但在 {events_dir} 未找到 {sid}_events.csv 或 .parquet")

        ev = ev.sort_values("time_s").reset_index(drop=True)
        n_ev = len(ev)
        if n_ev < 2:
            raise SystemExit(f"[error] {sid}：events_offset 模式需要至少 2 个事件。实际仅 {n_ev} 个。")

        offsets = m.get("inset", {})
        # 事件数与偏移数关系：只要求  (事件数-1) > 偏移数
        # 即，事件间隔应多于要应用偏移的窗口数；随后仅使用前 K=len(offsets) 个相邻事件间隔。
        W_all = n_ev - 1                         # 可形成的总窗口数
        K = len(offsets)                         # 研究者指定的偏移数量
        if K <= 0:
            raise SystemExit(f"[error] {sid}：events_offset.inset 未设置或为空。")
        if W_all <= K:
            raise SystemExit(
                f"[error] {sid}：事件数量不足：有 {n_ev} 个事件，可形成 {W_all} 个相邻窗；"
                f"但需要应用 {K} 个偏移（要求：事件窗数 > 偏移数）。"
            )

        # 逐窗按 1-based 窗号读取偏移；需要存在键 1..K
        d_list = []
        missing_keys = [j for j in range(1, K+1) if (j not in offsets and str(j) not in offsets)]
        if missing_keys:
            raise SystemExit(f"[error] {sid}：events_offset 缺少键 {missing_keys} 的偏移值（窗号从 1 开始连续编号）。")
        for j in range(1, K+1):
            v = offsets.get(j, offsets.get(str(j), 0.0))
            d_list.append(float(v or 0.0))

        # 仅取前 K 个相邻事件间隔
        for i in range(K):  # 仅取前 K 个相邻事件间隔
            s = float(ev.loc[i,   "time_s"])
            e = float(ev.loc[i+1, "time_s"])
            d = float(d_list[i])  # 两头各缩进 d 秒（来自已裁剪排序的偏移列表）
            s_adj = s + d
            e_adj = e - d
            # 落盘逻辑（反序/最小时长）沿用 add_win 的统一校验
            add_win(s_adj, e_adj, f"事件{i+1}内缩{offsets.get(str(i+1))}s")

    elif mode == "events_labeled_windows":
        # 基于 "start_event" 和 "end_event" 标签对来显式定义窗口，
        # 并对每个窗口应用 "inset" 内缩。
        # 这个模式解决了 events_offset 模式的两个问题：
        # 1. 它只切我们明确定义的窗口（如 baseline, induction, intervention），自动忽略中间的"垃圾"窗口（如问卷）。
        # 2. 它通过查找特定的 "end_event" 标签，可以正确处理点击组在干预窗内插入了多个 custom_event 的情况。
        
        m = wcfg["modes"].get("events_labeled_windows", {})
        
        # --- [加载事件文件的模板代码] ---
        # 事件文件路径优先使用本模式的 events_path
        events_dir = (DATA_DIR / m.get("events_path", "")).resolve()
        ev = load_events_for_sid(cfg, sid, prefer_dir=events_dir)
        if ev is None or ev.empty:
            raise SystemExit(f"[error] {sid}：events_labeled_windows 模式需要事件表，但在 {events_dir} 未找到 {sid}_events.csv 或 .parquet")
        # 确保事件按时间排序，这是我们"前向查找"逻辑的基础
        ev = ev.sort_values("time_s").reset_index(drop=True)

        # --- 核心逻辑 ---
        
        # 1. 从配置中获取我们想要处理的窗口定义列表
        windows_to_process = m.get("windows", [])
        if not windows_to_process or not isinstance(windows_to_process, list):
            raise SystemExit(f"[error] {sid}：events_labeled_windows.windows 配置未设置、为空或不是一个列表。")

        # 2. 初始化一个"搜索指针"
        # 这个指针记录了上一个窗口的结束时间，以确保我们总是在时间轴上"向前"搜索，
        # 这对于处理重名标签（如 custom_event）至关重要。
        last_search_time_s = -float('inf') 

        # 3. 按顺序遍历配置中定义的每一个窗口
        for win_def in windows_to_process:
            if not isinstance(win_def, dict):
                raise SystemExit(f"[error] {sid}：windows 列表中的项必须是字典，但找到了 {win_def}")

            # 4. 解析当前窗口的定义
            name = win_def.get("name")
            start_label = win_def.get("start_event")
            end_label = win_def.get("end_event")
            # 默认内缩为 0
            inset = float(win_def.get("inset", 0.0))

            # 验证配置是否完整
            if not all([name, start_label, end_label]):
                raise SystemExit(f"[error] {sid}：窗口定义 {win_def} 缺少 'name', 'start_event', 或 'end_event' 键。")

            # 5. 查找开始事件 (Start Event)
            # 核心逻辑：我们从 "last_search_time_s" 之后开始搜索，
            # 寻找第一个匹配 "start_label" 的事件。
            # start_search_mask = (ev["events"] == start_label) & (ev["time_s"] >= last_search_time_s)
            start_search_mask = (ev["events"] == start_label)
            start_row = ev[start_search_mask].iloc[0:1] # .iloc[0:1] 确保没找到时返回空DF而不报错

            if start_row.empty:
                raise SystemExit(f"[error] {sid}：(窗口 '{name}')：未能在 {last_search_time_s}s 后找到开始事件 '{start_label}'。")
            
            t_start = float(start_row["time_s"].iloc[0])

            # 6. 查找结束事件 (End Event)
            # 核心逻辑：我们从 "t_start" 之后开始搜索 (注意：不是 last_search_time_s)，
            # 寻找第一个匹配 "end_label" 的事件。
            # end_search_mask = (ev["events"] == end_label) & (ev["time_s"] > t_start)
            end_search_mask = (ev["events"] == end_label)
            end_row = ev[end_search_mask].iloc[0:1]

            if end_row.empty:
                raise SystemExit(f"[error] {sid}：(窗口 '{name}')：未能在 {t_start}s (事件 '{start_label}') 后找到结束事件 '{end_label}'。")

            t_end = float(end_row["time_s"].iloc[0])
            
            # 7. 应用偏移（内缩）并添加窗口
            s_adj = t_start + inset
            e_adj = t_end - inset
            
            # 使用一个清晰的名称调用 add_win
            win_name_with_info = f"{name} (内缩{inset}s)"
            add_win(s_adj, e_adj, win_name_with_info)
            
            # 8. 更新"搜索指针"
            # 我们将指针移动到当前窗口的结束时间，
            # 确保下一个循环会从这个时间点之后开始搜索。
            last_search_time_s = t_end

    elif mode == "events_single":
        m = wcfg["modes"]["events_single"]
        ev = load_events()
        #拿到 events.csv
        events_dir = (DATA_DIR / wcfg["modes"].get("events", {}).get("events_path", "")).resolve()
        if ev is None:
            raise SystemExit(f"[error] events_single 模式需要事件表，但在 {events_dir} 未找到 {sid}_events.csv 或 .parquet")
        
        # 锚点事件来自事件列表，去settings 填写。不可以为 None
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
            raise SystemExit("[error] events_sliding 需要事件文件，确认 {m} 文件夹下有该文件。")
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
            # subdivide 场景下的 single 语义：
            # start_s / end_s / win_len_s 都是**相对父窗起点 s0 的偏移秒数**。
            # 支持两种用法：[start_s, end_s] 或 [start_s, win_len_s]（二选一），
            # 若二者都省略，则默认 [start_s, 父窗结束 e0]。
            m = wcfg["modes"]["single"]
            off_start = m.get("start_s", None)
            off_end   = m.get("end_s", None)
            win_len   = m.get("win_len_s", None)

            if off_start is None:
                raise SystemExit("[error] subdivide/single 模式需提供 start_s（相对父窗起点的偏移秒数）")

            # 参数合法性检查：end_s 与 win_len_s 不能同时给
            if (off_end is not None) and (win_len is not None):
                raise SystemExit("[error] subdivide/single 模式参数冲突：end_s 与 win_len_s 只能二选一。")

            # 子窗起点：父窗起点 + start_s
            s = s0 + float(off_start)

            # 子窗终点：根据 end_s 或 win_len_s 决定，最后不得超出父窗 e0
            if off_end is not None:
                e = min(s0 + float(off_end), e0)
            elif win_len is not None:
                e = min(s + float(win_len), e0)
            else:
                # 两者都没设，默认切到父窗末尾
                e = e0

            # 最小时长检查
            min_win = PARAMS["min_window"]
            if (e - s) >= min_win:
                add_child(
                    s, e,
                    f"parent[w{parent_w_id:02d}] -> single[{fmt_s(s)},{fmt_s(e)}]"
                )
            else:
                print(
                    f"[warn] {sid}: 细分single窗口小于最小时长 {min_win}s 被拦截 s={s}, e={e}"
                )
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
        def _try_load(
            paths: list[Path],
            cols_map: dict[str, str] | None,
            tcol_name: str,
            required_cols: list[str] | None = None,
        ) -> tuple[pd.DataFrame | None, str]:
            """Try loading from candidate paths; return (df, tcol_name) on first success.

            Success criteria:
            - file exists and non-empty
            - required columns exist after renaming
            - time column can be parsed to numeric and has at least 1 valid row
            """
            for p in paths:
                df = read_table_if_exists(p, cols_map)
                if df is None or len(df) == 0:
                    continue

                # Validate required columns after renaming.
                req = list(required_cols) if required_cols else [tcol_name]
                missing = [c for c in req if c not in df.columns]
                if missing:
                    print(
                        f"[warn] {sid} {signal}: 文件 {p.name} 缺少列 {missing}，已跳过。现有列: {list(df.columns)}"
                    )
                    continue

                # Coerce time to numeric for reliable window slicing.
                df[tcol_name] = pd.to_numeric(df[tcol_name], errors="coerce")
                df = df.dropna(subset=[tcol_name]).sort_values(tcol_name).reset_index(drop=True)
                if len(df) == 0:
                    continue

                return df, tcol_name

            return None, ""

        if signal == "rr":
            paths = [cfg["paths"]["rr"] / f"{sid}_rr.csv", cfg["paths"]["rr"] / f"{sid}_rr.parquet"]
            return _try_load(paths, {"t_s":"t_s", "rr_ms":"rr_ms"}, "t_s", required_cols=["t_s","rr_ms"])
        if signal == "hr":
            paths = [cfg["paths"]["other"] / f"{sid}_hr.csv", cfg["paths"]["other"] / f"{sid}_hr.parquet"]
            # SCHEMA['hr'] is already t_s,bpm, so no renaming needed
            return _try_load(paths, None, "t_s", required_cols=["t_s","bpm"])
        if signal == "resp":
            paths = [cfg["paths"]["other"] / f"{sid}_resp.csv", cfg["paths"]["other"] / f"{sid}_resp.parquet"]
            return _try_load(paths, {SCHEMA["resp"]["t"]:"time_s", SCHEMA["resp"]["v"]:"value"}, "time_s", required_cols=["time_s","value"])
        if signal == "ecg":
            paths = [cfg["paths"]["other"] / f"{sid}_ecg.csv", cfg["paths"]["other"] / f"{sid}_ecg.parquet"]
            return _try_load(paths, {SCHEMA["ecg"]["t"]:"time_s", SCHEMA["ecg"]["v"]:"value"}, "time_s", required_cols=["time_s","value"])
        if signal == "acc":
            paths = [cfg["paths"]["other"] / f"{sid}_acc.csv", cfg["paths"]["other"] / f"{sid}_acc.parquet"]
            return _try_load(
                paths,
                {SCHEMA["acc"]["t"]:"time_s", SCHEMA["acc"]["vx"]:"value_x", SCHEMA["acc"]["vy"]:"value_y", SCHEMA["acc"]["vz"]:"value_z"},
                "time_s",
                required_cols=["time_s","value_x","value_y","value_z"],
            )
        if signal == "events":
            paths = [cfg["paths"]["other"] / f"{sid}_events.csv", cfg["paths"]["other"] / f"{sid}_events.parquet"]
            return _try_load(paths, {SCHEMA["events"]["t"]:"time_s", SCHEMA["events"]["label"]:"events"}, "time_s", required_cols=["time_s","events"])
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
        if expect > 0 and n_saved == 0:
            try:
                sig_min = float(df_sig[tcol].min())
                sig_max = float(df_sig[tcol].max())
                win_min = float(windows_df["t_start_s"].min())
                win_max = float(windows_df["t_end_s"].max())
                print(
                    f"[hint] {sid} {signal}: 全部窗口无数据。"
                    f"信号时间范围=[{fmt_s(sig_min)}~{fmt_s(sig_max)}]，"
                    f"窗口范围=[{fmt_s(win_min)}~{fmt_s(win_max)}]。"
                    f"请检查该信号的时间基准是否与RR/事件一致，或是否存在单位不一致（ms vs s）。"
                )
            except Exception:
                pass
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
        # 按被试逐一列举全部窗口
        for sid_i in sorted(df['subject_id'].unique()):
            sub_i = df[df['subject_id']==sid_i]
            f.write(f"被试：{sid_i}（{len(sub_i)} 个窗）\n")
            for _, r in sub_i.iterrows():
                dur = float(r['t_end_s']) - float(r['t_start_s'])
                f.write(
                    f"  · L{int(r['level'])}:w{int(r['w_id']):02d}  "
                    f"[{fmt_s(r['t_start_s'])} ~ {fmt_s(r['t_end_s'])}]  "
                    f"时长={fmt_s(dur)}  "
                    f"parent=L{int(r['parent_level'])}:w{int(r['parent_w_id']):02d}  "
                    f"{r['meaning']}\n"
                )
            f.write("\n")
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