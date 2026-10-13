"""Render metrics.csv into a grid of line charts (dashboard.png)."""

from __future__ import annotations

import csv
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Panel:
    """One chart: the metrics.csv columns to plot together."""

    title: str
    columns: tuple[str, ...]
    smooth: bool = True


def _read_columns(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = list(reader)
    columns: dict[str, np.ndarray] = {}
    for index, name in enumerate(header):
        values = np.full(len(rows), np.nan)
        for row_index, row in enumerate(rows):
            if index < len(row) and row[index] not in ("", "nan"):
                try:
                    values[row_index] = float(row[index])
                except ValueError:
                    pass
        columns[name] = values
    return columns


def _smooth(values: np.ndarray) -> np.ndarray:
    window = max(1, min(200, len(values) // 50))
    if window <= 1:
        return values
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    kernel = np.ones(window)
    total = np.convolve(filled, kernel, mode="same")
    count = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def render_dashboard(metrics: Path, output: Path, panels: list[Panel], *, title: str) -> bool:
    """Write ``output``; returns False when there is nothing to plot yet."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not Path(metrics).is_file():
        return False
    data = _read_columns(metrics)
    if "iteration" not in data or not len(data["iteration"]):
        return False
    shown = [panel for panel in panels if any(column in data for column in panel.columns)]
    if not shown:
        return False
    columns = 3
    rows = math.ceil(len(shown) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.4 * rows), squeeze=False)
    x = data["iteration"]
    for axis, panel in zip(axes.flat, shown):
        for column in panel.columns:
            if column not in data:
                continue
            y = data[column]
            if np.all(np.isnan(y)):
                continue
            if panel.smooth and len(y) > 50:
                axis.plot(x, y, alpha=0.15, linewidth=0.6)
                axis.plot(x, _smooth(y), linewidth=1.2, label=column, color=axis.lines[-1].get_color())
            else:
                axis.plot(x, y, linewidth=1.0, label=column)
        axis.set_title(panel.title, fontsize=10)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, loc="best")
    for axis in list(axes.flat)[len(shown) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output = Path(output)
    temporary = output.with_name(f".{output.name}.tmp.png")
    figure.savefig(temporary, dpi=110)
    plt.close(figure)
    temporary.replace(output)
    return True


class DashboardRefresher:
    """Re-render every ``every`` iterations on a background thread."""

    def __init__(self, metrics: Path, output: Path, panels: list[Panel], title: str, every: int) -> None:
        self.metrics, self.output, self.panels, self.title, self.every = (
            metrics, output, panels, title, every,
        )
        self._thread: threading.Thread | None = None

    def maybe_refresh(self, iteration: int, *, force: bool = False) -> None:
        if not force and (self.every <= 0 or iteration % self.every):
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=render_dashboard,
            args=(self.metrics, self.output, self.panels),
            kwargs={"title": self.title},
            daemon=True,
        )
        self._thread.start()

    def finish(self) -> None:
        if self._thread is not None:
            self._thread.join()
