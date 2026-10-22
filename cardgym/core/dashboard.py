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
    """One chart: the metrics.csv columns to plot together.

    ``log`` draws a log y-axis (and smooths in log space, so a rare spike
    moves the trend by a factor rather than by its size). ``top`` caps the
    y-axis: values above it run off the chart. ``robust`` fits the y-axis to
    the 0.5th-99.5th percentiles of the plotted values instead of their range.
    ``styles`` gives each column a (color index, line style), e.g. one color
    per head and a style per quantity; empty uses matplotlib's cycle.
    """

    title: str
    columns: tuple[str, ...]
    smooth: bool = True
    log: bool = False
    top: float | None = None
    robust: bool = False
    styles: tuple[tuple[int, str], ...] = ()


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


def _smooth(values: np.ndarray, *, log: bool = False) -> np.ndarray:
    window = max(1, min(200, len(values) // 50))
    if window <= 1:
        return values
    if log:
        with np.errstate(invalid="ignore", divide="ignore"):
            logs = np.where(np.isfinite(values) & (values > 0), np.log(values), np.nan)
        return np.exp(_smooth(logs))
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    kernel = np.ones(window)
    total = np.convolve(filled, kernel, mode="same")
    count = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _limit(axis, panel: Panel, plotted: list[np.ndarray]) -> None:
    """Apply the panel's log scale and y-axis caps."""

    values = np.concatenate(plotted) if plotted else np.empty(0)
    values = values[np.isfinite(values)]
    if panel.log:
        values = values[values > 0]
        if not len(values):
            return
        axis.set_yscale("log")
    if panel.robust and len(values):
        low, high = np.percentile(values, [0.5, 99.5])
        if panel.log:
            axis.set_ylim(low / 1.5, high * 1.5)
        else:
            margin = 0.05 * (high - low) or 0.05 * abs(high) or 1.0
            axis.set_ylim(low - margin, high + margin)
    if panel.top is not None:
        bottom, top = axis.get_ylim()
        if top > panel.top:
            axis.set_ylim(bottom, panel.top)


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
        plotted = []
        for index, column in enumerate(panel.columns):
            if column not in data:
                continue
            y = data[column]
            if np.all(np.isnan(y)):
                continue
            plotted.append(y)
            style = {}
            if index < len(panel.styles):
                color, linestyle = panel.styles[index]
                style = {"color": f"C{color}", "linestyle": linestyle}
            if panel.smooth and len(y) > 50:
                axis.plot(x, y, alpha=0.15, linewidth=0.6, **style)
                style.setdefault("color", axis.lines[-1].get_color())
                axis.plot(x, _smooth(y, log=panel.log), linewidth=1.2, label=column, **style)
            else:
                axis.plot(x, y, linewidth=1.0, label=column, **style)
        _limit(axis, panel, plotted)
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
