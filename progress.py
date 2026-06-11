from __future__ import annotations

import sys
import time
from datetime import datetime


def log(msg: str, *, rank: int | None = 0, flush: bool = True) -> None:
    """带时间戳的进度日志；rank!=0 时不打印（用于 DDP）。"""
    if rank is not None and rank != 0:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr if False else sys.stdout, flush=flush)


class ProgressTracker:
    def __init__(self, total: int, desc: str = "", rank: int = 0) -> None:
        self.total = max(1, total)
        self.desc = desc
        self.rank = rank
        self.current = 0
        self._t0 = time.time()

    def update(self, n: int = 1, extra: str = "") -> None:
        self.current += n
        if self.rank != 0:
            return
        pct = 100.0 * self.current / self.total
        elapsed = time.time() - self._t0
        eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0.0
        suffix = f" | {extra}" if extra else ""
        log(f"{self.desc} [{self.current}/{self.total}] {pct:.1f}% | elapsed {elapsed:.0f}s | eta {eta:.0f}s{suffix}", rank=0)

    def done(self, msg: str = "") -> None:
        if self.rank == 0:
            log(f"{self.desc} 完成 ({self.current}/{self.total}){': ' + msg if msg else ''}", rank=0)
