# scripts/loaders/fantasia_loader.py
# 使用 Fantasia loader 的注意事项:
#
# 1) 本地优先还是在线优先？
#    - 代码会先在本地 RAW_CACHE_DIR 查找 **官方WFDB原始文件**（成对的 .hea/.dat）。
#      如果找到，就本地读取；找不到才会去 PhysioNet 远端（pn_dir="fantasia"）下载读取。
#
# 2) 本地文件该放哪里、叫什么名？
#    - 目录：settings.RAW_CACHE_DIR，默认是 data/raw/physionet/fantasia/
#    - 文件名必须是 官方基名：
#        f1y01.hea 与 f1y01.dat
#        f1o01.hea 与 f1o01.dat
#      也就是说：<记录名>.hea 与 <记录名>.dat 必须同目录同名成对存在。
#    - 只要 .hea 存在而 .dat 缺失，WFDB 也读不动；两个都要有。
#
# 3) 别把本地路径传给 pn_dir
#    - 本地读取应传 **完整本地路径+记录名** 给 rdrecord：wfdb.rdrecord(str(local_dir / rec))
#    - 不要把本地目录当 pn_dir 传，否则 WFDB 会把它当“远端子目录”拼 URL，报 404。
#
# 4) 缓存的 parquet 是额外福利，不是必需品
#    - 本 loader 会把每条记录的 ecg/resp 另存为：
#        data/raw/physionet/fantasia/<记录名>_ecg.parquet
#        data/raw/physionet/fantasia/<记录名>_resp.parquet
#      这是为了二次读取快一些；删了也不影响“用 .hea/.dat 读原始”的主流程。
#    - 注意：仅有 parquet 而无 .hea/.dat 时，loader 仍会尝试在线读取。
#
# 5) 记录名单怎么控制？
#    - 在 settings.DATASETS[*]["records"] 指定列表（如 ["f1y01","f1o01"]）只读这些；
#      留空或不写该键，则自动读取 **全库**（优先本地有的，剩下走远端）。
#
# 6) 目录结构示例（推荐）：
#    data/
#      raw/
#        physionet/
#          fantasia/
#            f1y01.hea
#            f1y01.dat
#            f1o01.hea
#            f1o01.dat
#            f1y01_ecg.parquet      # 由 loader 自动生成，可有可无
#            f1y01_resp.parquet     # 由 loader 自动生成，可有可无
#
# 7) 常见报错速查：
#    - NetFileNotFoundError (404)：多半是把本地目录误传给了 pn_dir；或 .hea/.dat 缺一半。
#    - 读到的通道没有 ECG/Resp：检查记录是否来自 Fantasia，或通道名是否与官方一致。

import pandas as pd
import numpy as np
import wfdb

# --- project-root bootstrap ---
import sys
from pathlib import Path
_p = Path(__file__).resolve()
for _ in range(6):  # 最多向上爬6层
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent))
        break
    _p = _p.parent
# --- end bootstrap ---
from settings import VERBOSE, CACHE_RAW, RAW_CACHE_DIR

# —— 官方完整名单获取（失败就退回静态列表） ——
_STATIC_RECORDS = [
    # elderly f1o**, f2o**
    "f1o01","f1o02","f1o03","f1o04","f1o05","f1o06","f1o07","f1o08","f1o09","f1o10",
    "f2o01","f2o02","f2o03","f2o04","f2o05","f2o06","f2o07","f2o08","f2o09","f2o10",
    # young f1y**, f2y**
    "f1y01","f1y02","f1y03","f1y04","f1y05","f1y06","f1y07","f1y08","f1y09","f1y10",
    "f2y01","f2y02","f2y03","f2y04","f2y05","f2y06","f2y07","f2y08","f2y09","f2y10",
]

def _list_all_records():
    try:
        recs = wfdb.get_record_list('fantasia')  # 官方 RECORDS 索引
        recs = [r.strip() for r in recs if r.strip()]
        if len(recs) >= 40:
            if VERBOSE: print(f"[fantasia] found {len(recs)} records from PhysioNet index.")
            return recs
    except Exception as e:
        if VERBOSE: print(f"[fantasia] get_record_list failed, fallback to static list. ({e})")
    return _STATIC_RECORDS

def _rdrecord_local_or_remote(rec: str):
    # 本地优先：如果 RAW_CACHE_DIR 下有 f1y01.hea，就从本地读
    local_dir = RAW_CACHE_DIR
    hea = local_dir / f"{rec}.hea"
    if hea.exists():
        if VERBOSE:
            print(f"          local WFDB → {hea}")
        # 关键修正：本地读不要用 pn_dir，直接给“完整路径/记录名”
        # WFDB 会自动去找同目录下的 .dat/.hea
        return wfdb.rdrecord(str(local_dir / rec))
    # 否则走在线目录（pn_dir 用数据库名）
    return wfdb.rdrecord(rec, pn_dir="fantasia")

def load(dataset_cfg: dict, data_dir: Path) -> pd.DataFrame:
    # 记录列表：settings 里指定的，否则全库
    records = dataset_cfg.get("records")
    if not records:
        records = _list_all_records()

    age_map = dataset_cfg.get("age_group_map", {})
    if CACHE_RAW:
        RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dfs = []
    n = len(records)
    for i, rec in enumerate(records, 1):
        if VERBOSE:
            print(f"[fantasia] [{i}/{n}] reading '{rec}' ...", flush=True)
        record = _rdrecord_local_or_remote(rec)
        fs = float(record.fs)
        t = np.arange(record.sig_len) / fs

        signals = {}
        for ch_idx, ch_name in enumerate(record.sig_name):
            name = ch_name.upper()
            if "ECG" in name:
                signals["ecg"] = record.p_signal[:, ch_idx]
            elif "RESP" in name:
                signals["resp"] = record.p_signal[:, ch_idx]

        age_group = _infer_age_group(rec, age_map)

        for sig, arr in signals.items():
            df = pd.DataFrame({
                "subject_id": rec,
                "age_group": age_group,
                "signal": sig,
                "fs_hz": fs,
                "time_s": t,
                "value": arr.astype(float),
            })
            dfs.append(df)

            if CACHE_RAW:
                out_path = RAW_CACHE_DIR / f"{rec}_{sig}.parquet"
                df.to_parquet(out_path, index=False)
                if VERBOSE:
                    dur = t[-1] if len(t) else 0
                    print(f"          cached → {out_path}  (fs={fs:.0f} Hz, dur≈{dur:.0f}s, n={len(df)})", flush=True)

        if VERBOSE:
            print(f"[fantasia] [{i}/{n}] done '{rec}'.", flush=True)

    return pd.concat(dfs, ignore_index=True)

def _infer_age_group(rec: str, mapping: dict) -> str:
    token = rec[2] if len(rec) >= 3 else ""
    return mapping.get(token, "unknown")