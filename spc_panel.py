from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from matplotlib.figure import Figure


class SPCPanel:
    """Framework-agnostic matplotlib I-MR panel."""

    DATA_COLOR = "#185FA5"
    OOC_COLOR = "#E24B4A"
    LIMIT_COLOR = "#D85A30"
    CENTER_COLOR = "#5F5E5A"
    COMFORT_COLOR = "#97C459"

    def __init__(self) -> None:
        self.fig = Figure(figsize=(10, 6), tight_layout=True)
        self.ax_i, self.ax_mr = self.fig.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 2]},
        )

        self.ax_i.set_ylabel("Temp (°C)")
        self.ax_mr.set_ylabel("MR")
        self.ax_mr.set_xlabel("Sample")

        self.ax_i.grid(True, alpha=0.25)
        self.ax_mr.grid(True, alpha=0.25)

        self.comfort_band = self.ax_i.axhspan(
            22,
            25,
            color=self.COMFORT_COLOR,
            alpha=0.18,
            zorder=0,
        )

        (self.line_i,) = self.ax_i.plot([], [], color=self.DATA_COLOR, marker="o", ms=4, lw=1.6)
        (self.line_mr,) = self.ax_mr.plot([], [], color=self.DATA_COLOR, marker="o", ms=4, lw=1.4)

        (self.ucl_i_line,) = self.ax_i.plot([], [], "--", color=self.LIMIT_COLOR, lw=1.5)
        (self.cl_i_line,) = self.ax_i.plot([], [], "-", color=self.CENTER_COLOR, lw=1.2)
        (self.lcl_i_line,) = self.ax_i.plot([], [], "--", color=self.LIMIT_COLOR, lw=1.5)

        (self.ucl_mr_line,) = self.ax_mr.plot([], [], "--", color=self.LIMIT_COLOR, lw=1.5)
        (self.cl_mr_line,) = self.ax_mr.plot([], [], "-", color=self.CENTER_COLOR, lw=1.2)

        self.ooc_i = self.ax_i.scatter([], [], s=55, color=self.OOC_COLOR, zorder=5)
        self.ooc_mr = self.ax_mr.scatter([], [], s=45, color=self.OOC_COLOR, zorder=5)

        self.ax_i.set_title("Individuals (I) Chart")
        self.ax_mr.set_title("Moving Range (MR) Chart")
        self.fig.suptitle("SPC Temperature Monitor", fontsize=13, fontweight="bold")

    @staticmethod
    def _line_xy(x: List[float], y: float) -> Tuple[List[float], List[float]]:
        if not x:
            return [], []
        return [x[0], x[-1]], [y, y]

    def update(self, snapshot: Dict[str, object]) -> None:
        x = list(snapshot.get("indices", []))
        y = list(snapshot.get("values", []))
        mr = list(snapshot.get("mr", []))

        mr_x = x[1:] if len(x) > 1 else []

        self.line_i.set_data(x, y)
        self.line_mr.set_data(mr_x, mr)

        ucl_i = snapshot.get("ucl_i")
        lcl_i = snapshot.get("lcl_i")
        cl_i = snapshot.get("target")
        ucl_mr = snapshot.get("ucl_mr")
        cl_mr = snapshot.get("mr_bar")

        self.ucl_i_line.set_data(*self._line_xy(x, ucl_i)) if x and ucl_i is not None else self.ucl_i_line.set_data([], [])
        self.lcl_i_line.set_data(*self._line_xy(x, lcl_i)) if x and lcl_i is not None else self.lcl_i_line.set_data([], [])
        self.cl_i_line.set_data(*self._line_xy(x, cl_i)) if x and cl_i is not None else self.cl_i_line.set_data([], [])
        self.ucl_mr_line.set_data(*self._line_xy(mr_x, ucl_mr)) if mr_x and ucl_mr is not None else self.ucl_mr_line.set_data([], [])
        self.cl_mr_line.set_data(*self._line_xy(mr_x, cl_mr)) if mr_x and cl_mr is not None else self.cl_mr_line.set_data([], [])

        comfort_low = float(snapshot.get("comfort_low", 22))
        comfort_high = float(snapshot.get("comfort_high", 25))
        self.comfort_band.remove()
        self.comfort_band = self.ax_i.axhspan(
            comfort_low,
            comfort_high,
            color=self.COMFORT_COLOR,
            alpha=0.18,
            zorder=0,
        )

        alarm_ix = list(snapshot.get("alarm_indices", []))
        alarm_vals = list(snapshot.get("alarm_values", []))
        if alarm_ix and alarm_vals:
            offsets = np.column_stack([alarm_ix, alarm_vals])
        else:
            offsets = np.empty((0, 2))
        self.ooc_i.set_offsets(offsets)

        mr_alarm_ix = list(snapshot.get("mr_alarm_indices", []))
        mr_alarm_vals = list(snapshot.get("mr_alarm_values", []))
        if mr_alarm_ix and mr_alarm_vals:
            mr_offsets = np.column_stack([mr_alarm_ix, mr_alarm_vals])
        else:
            mr_offsets = np.empty((0, 2))
        self.ooc_mr.set_offsets(mr_offsets)

        if x:
            xmin = max(1, x[0])
            xmax = x[-1] if x[-1] > xmin else xmin + 1
            self.ax_i.set_xlim(xmin, xmax)
            self.ax_mr.set_xlim(xmin, xmax)

        all_i_y = [v for v in y]
        for lim in (ucl_i, lcl_i, cl_i, comfort_low, comfort_high):
            if lim is not None:
                all_i_y.append(float(lim))
        self.ax_i.set_ylim(20, 28)

        all_mr_y = [v for v in mr]
        for lim in (ucl_mr, cl_mr):
            if lim is not None:
                all_mr_y.append(float(lim))
        if all_mr_y:
            ymin = 0
            ymax = max(all_mr_y)
            pad = max(0.05, 0.15 * ymax) if ymax > 0 else 0.2
            self.ax_mr.set_ylim(ymin, ymax + pad)

        current = snapshot.get("current")
        mean = snapshot.get("mean")
        sigma_hat = snapshot.get("sigma_hat")
        mr_bar = snapshot.get("mr_bar")
        alarm_count = len(snapshot.get("alarms", []))
        calibrated = snapshot.get("calibrated")

        self.fig.suptitle(
            " | ".join(
                [
                    f"Current T: {current:.2f} °C" if current is not None else "Current T: --",
                    f"Mean: {mean:.2f}" if mean is not None else "Mean: --",
                    f"σ̂: {sigma_hat:.4f}" if sigma_hat is not None else "σ̂: --",
                    f"MR̄: {mr_bar:.4f}" if mr_bar is not None else "MR̄: --",
                    f"Alarms: {alarm_count}",
                    "Calibrated" if calibrated else "Phase-I",
                ]
            ),
            fontsize=12,
            fontweight="bold",
        )

    def redraw(self) -> None:
        canvas = getattr(self.fig, "canvas", None)
        if canvas is not None:
            draw_idle = getattr(canvas, "draw_idle", None)
            if callable(draw_idle):
                draw_idle()
            else:
                canvas.draw()
