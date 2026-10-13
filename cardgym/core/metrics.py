"""Append-only CSV metrics with resume-safe truncation.

The header is the union of every column seen so far. A row with a new
column (a code change between restarts, say) rewrites the file once with the
widened header; older rows read the new column as empty.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .io import atomic_write_text


class MetricsLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._header: list[str] | None = None

    def header(self) -> list[str]:
        if self._header is None:
            if self.path.is_file() and self.path.stat().st_size:
                with self.path.open(newline="") as handle:
                    self._header = next(csv.reader(handle), [])
            else:
                self._header = []
        return self._header

    def append(self, row: dict[str, Any]) -> None:
        header = self.header()
        missing = [key for key in row if key not in header]
        if missing:
            rows = self.read() if header else []
            self._header = header + missing
            self._rewrite(rows)
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._header, lineterminator="\n")
            writer.writerow({key: _format(value) for key, value in row.items()})

    def read(self) -> list[dict[str, str]]:
        if not self.path.is_file():
            return []
        with self.path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def last_value(self, column: str, default: float = 0.0) -> float:
        rows = self.read()
        if not rows or not rows[-1].get(column):
            return default
        return float(rows[-1][column])

    def truncate_after(self, iteration: int) -> int:
        """Drop rows newer than ``iteration``; returns how many were dropped."""

        rows = self.read()
        kept = [row for row in rows if int(float(row.get("iteration", 0) or 0)) <= iteration]
        if len(kept) != len(rows):
            self._rewrite(kept)
        return len(rows) - len(kept)

    def _rewrite(self, rows: list[dict[str, str]]) -> None:
        header = self.header()
        lines: list[str] = []

        class _Sink:
            def write(self, text: str) -> None:
                lines.append(text)

        writer = csv.DictWriter(_Sink(), fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})
        atomic_write_text(self.path, "".join(lines))


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return format(value, ".7g")
    return str(value)
