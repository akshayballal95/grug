"""A progress bar for paid runs: how far along, how fast, what it has cost.

Written here rather than pulled in as a dependency because it needs to report
spend alongside position, which no general-purpose bar knows about.
"""

from __future__ import annotations

import shutil
import sys
import time

__all__ = ["ProgressBar"]


def _clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


class ProgressBar:
    """Single-line bar on stderr, so stdout stays pipeable."""

    def __init__(
        self,
        total: int,
        *,
        label: str = "",
        stream=None,
        min_interval: float = 0.25,
        log_interval: float = 20.0,
    ):
        self.total = max(1, total)
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.min_interval = min_interval
        self.log_interval = log_interval
        self.start = time.monotonic()
        self._last = 0.0
        self._done = 0
        self.enabled = getattr(self.stream, "isatty", lambda: False)()

    def update(self, done: int, *, cost: float = 0.0, note: str = "", force: bool = False) -> None:
        self._done = done
        now = time.monotonic()
        interval = self.min_interval if self.enabled else self.log_interval
        if not force and now - self._last < interval:
            return
        self._last = now
        line = self._render(done, cost, note)
        # Redraw in place on a terminal; append a line when piped to a log, so a
        # background run still shows movement instead of nothing until it ends.
        self.stream.write("\r\x1b[2K" + line if self.enabled else line + "\n")
        self.stream.flush()

    def _render(self, done: int, cost: float, note: str) -> str:
        frac = min(1.0, done / self.total)
        elapsed = time.monotonic() - self.start
        rate = done / elapsed if elapsed > 0 and done else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0

        right = f" {done}/{self.total}"
        if rate:
            right += f"  {rate * 60:.0f}/min  eta {_clock(eta)}"
        if cost:
            right += f"  ${cost:.2f}"
            if done:
                right += f"  (${cost / done * self.total:.2f} projected)"
        if note:
            right += f"  {note}"

        head = f"{self.label} " if self.label else ""
        width = shutil.get_terminal_size((100, 24)).columns
        bar_width = max(8, width - len(head) - len(right) - 8)
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        return f"{head}[{bar}] {frac * 100:3.0f}%{right}"

    def close(self, *, cost: float = 0.0, note: str = "") -> None:
        self.update(self._done, cost=cost, note=note, force=True)
        if self.enabled:
            self.stream.write("\n")
        self.stream.flush()
