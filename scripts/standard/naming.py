# scripts/standard/naming.py
from __future__ import annotations
import re
from typing import Optional

def detect_signal(name: str, aliases: dict[str, list[str]] | None = None) -> Optional[str]:
    """从文件名识别信号类型；aliases 为空时尝试从 settings 导入默认映射。"""
    if aliases is None:
        try:
            from settings import SIGNAL_ALIASES as _ALIASES
            aliases = _ALIASES
        except Exception:
            aliases = {}
    low = name.lower()
    for sig, keys in aliases.items():
        if any(k in low for k in keys):
            return sig
    return None

def infer_sid(name: str, pattern: str | None = None) -> Optional[str]:
    """
    从文件名推断统一 SID：
      - 已是 P001S001T001R001 → 原样大写返回
      - Fantasia: f1o01/f1y02 → 原样返回
      - BIDS-ish: sub-*, ses-*, task-*, run-* → 组装为 P{p}S{s}T{t}R{r}
      - 其他：抓第一个数字当 p，其余补 001
    可选 pattern 例如 "P{p:03d}S{s:03d}T{t:03d}R{r:03d}"。
    """
    low = name.lower()

    # 已是规范 P###S###T###R###
    m = re.search(r'(p\d{3}s\d{3}t\d{3}r\d{3})', low)
    if m:
        return m.group(1).upper()

    # Fantasia ID
    m = re.match(r'^(f[12][oy]\d{2})', low)
    if m:
        return m.group(1)

    # BIDS-ish
    sub = re.search(r'sub-([a-z0-9]+)', low)
    ses = re.search(r'ses-([a-z0-9]+)', low)
    task = re.search(r'task-([a-z0-9]+)', low)
    run = re.search(r'run-([a-z0-9]+)', low)

    def _num(tok, default=1):
        if not tok: return default
        m = re.search(r'(\d+)', tok)
        return int(m.group(1)) if m else default

    if any([sub, ses, task, run]):
        p = _num(sub.group(1) if sub else None, 1)
        s = _num(ses.group(1) if ses else None, 1)
        t = _num(task.group(1) if task else None, 1)  # 非数字 task → 001
        r = _num(run.group(1) if run else None, 1)
        if pattern:
            try:
                return pattern.format(p=p, s=s, t=t, r=r)
            except Exception:
                # pattern 写坏了就退回默认格式
                return f"P{p:03d}S{s:03d}T{t:03d}R{r:03d}"
        return f"P{p:03d}S{s:03d}T{t:03d}R{r:03d}"

    # 最保守兜底：抓第一个数字当 p
    digits = re.findall(r'(\d+)', low)
    if digits:
        p = int(digits[0])
        if pattern:
            try:
                return pattern.format(p=p, s=1, t=1, r=1)
            except Exception:
                pass
        return f"P{p:03d}S001T001R001"

    return None