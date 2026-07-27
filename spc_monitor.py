from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class IMRMonitor:
    """I-MR SPC monitor with optional Phase-I auto calibration.

    النسخة هذه مبسطة:
    - الألارم في مخطط الأفراد I يكون فقط إذا النقطة خارج UCL / LCL
    - بدون قواعد WECO
    - MR chart يبقى محسوبًا، ويمكن تسجيل إنذار إذا تعدى UCL_MR
    """

    D2 = 1.128
    D4 = 3.267

    def __init__(
        self,
        target: float,
        ucl_i: Optional[float],
        lcl_i: Optional[float],
        ucl_mr: Optional[float],
        comfort_low: float = 22,
        comfort_high: float = 25,
        window: int = 300,
        auto_calibrate: bool = False,
        phase1_size: int = 25,
    ) -> None:
        self.initial_target = float(target)
        self.initial_ucl_i = None if ucl_i is None else float(ucl_i)
        self.initial_lcl_i = None if lcl_i is None else float(lcl_i)
        self.initial_ucl_mr = None if ucl_mr is None else float(ucl_mr)
        self.comfort_low = float(comfort_low)
        self.comfort_high = float(comfort_high)
        self.window = int(window)
        self.auto_calibrate = bool(auto_calibrate)
        self.phase1_size = int(phase1_size)

        self.reset()

    def reset(self) -> None:
        self.values: Deque[float] = deque(maxlen=self.window)
        self.mr_values: Deque[float] = deque(maxlen=self.window)
        self.indices: Deque[int] = deque(maxlen=self.window)
        self.phase1_buffer: List[float] = []

        self.alarm_log: List[Tuple[int, float, str]] = []
        self.alarm_indices: set[int] = set()

        self.count = 0
        self.target = self.initial_target
        self.ucl_i = self.initial_ucl_i
        self.lcl_i = self.initial_lcl_i
        self.ucl_mr = self.initial_ucl_mr

        self.sigma_hat: Optional[float] = None
        self.mr_bar: Optional[float] = None
        self.mean: Optional[float] = None

        self.calibrated = not self.auto_calibrate

        if (
            self.ucl_i is not None
            and self.lcl_i is not None
            and self.target is not None
        ):
            self.sigma_hat = (self.ucl_i - self.target) / 3.0

        if self.ucl_mr is not None:
            self.mr_bar = self.ucl_mr / self.D4

    def _current_sigma(self) -> Optional[float]:
        if self.sigma_hat is not None:
            return float(self.sigma_hat)

        if len(self.mr_values) > 0:
            return float(np.mean(self.mr_values) / self.D2)

        return None

    def _log_alarm(self, idx: int, value: float, reason: str) -> None:
        if (idx, reason) not in {(a[0], a[2]) for a in self.alarm_log}:
            self.alarm_log.append((idx, float(value), reason))
        self.alarm_indices.add(idx)

    def _calibrate_from_phase1(self) -> None:
        arr = np.asarray(self.phase1_buffer, dtype=float)
        mr = np.abs(np.diff(arr))

        mr_bar = float(np.mean(mr)) if mr.size else 0.0
        sigma_hat = float(mr_bar / self.D2) if mr.size else 0.0
        mean = float(np.mean(arr)) if arr.size else self.initial_target

        self.target = mean
        self.mean = mean
        self.mr_bar = mr_bar
        self.sigma_hat = sigma_hat
        self.ucl_i = mean + 3.0 * sigma_hat
        self.lcl_i = mean - 3.0 * sigma_hat
        self.ucl_mr = self.D4 * mr_bar
        self.calibrated = True

    def _evaluate_rules(self) -> List[str]:
        """نسخة مبسطة: فقط فحص الخروج عن حدود I-chart و MR-chart."""
        reasons: List[str] = []

        if not self.calibrated:
            return reasons

        if not self.values or self.ucl_i is None or self.lcl_i is None:
            return reasons

        vals = np.asarray(self.values, dtype=float)
        idxs = list(self.indices)

        x = float(vals[-1])
        idx = idxs[-1]

        # Individuals chart
        if x > self.ucl_i:
            reasons.append("Above UCL")
        elif x < self.lcl_i:
            reasons.append("Below LCL")

        # Moving Range chart
        if self.mr_values and self.ucl_mr is not None:
            mr_last = float(self.mr_values[-1])
            if mr_last > self.ucl_mr:
                reasons.append("MR above UCL")

        for reason in reasons:
            self._log_alarm(idx, x, reason)

        return reasons

    def add(self, value: float) -> Dict[str, object]:
        x = float(value)
        self.count += 1
        idx = self.count

        prev = self.values[-1] if self.values else None

        self.values.append(x)
        self.indices.append(idx)

        mr = None
        if prev is not None:
            mr = abs(x - prev)
            self.mr_values.append(float(mr))

        # Auto-calibration phase
        if self.auto_calibrate and not self.calibrated:
            self.phase1_buffer.append(x)
            self.mean = float(np.mean(self.phase1_buffer))

            if len(self.phase1_buffer) >= 2:
                phase1_mr = np.abs(np.diff(np.asarray(self.phase1_buffer, dtype=float)))
                self.mr_bar = float(np.mean(phase1_mr))
                self.sigma_hat = float(self.mr_bar / self.D2)

            if len(self.phase1_buffer) >= self.phase1_size:
                self._calibrate_from_phase1()

        # Live stats update
        if self.values:
            self.mean = float(np.mean(np.asarray(self.values, dtype=float)))

        if self.mr_values:
            self.mr_bar = float(np.mean(np.asarray(self.mr_values, dtype=float)))
            if self.sigma_hat is None or self.calibrated:
                self.sigma_hat = float(self.mr_bar / self.D2)

        reasons = self._evaluate_rules() if self.calibrated else []
        ooc = bool(reasons)

        return {
            "index": idx,
            "value": x,
            "mr": mr,
            "ooc": ooc,
            "reasons": reasons,
            "calibrated": self.calibrated,
        }

    def get_snapshot(self) -> Dict[str, object]:
        values = list(self.values)
        mr = list(self.mr_values)
        indices = list(self.indices)

        # نقاط الألارم الخاصة بـ I-chart
        alarm_points = [i for i in indices if i in self.alarm_indices]
        alarm_lookup = {idx: val for idx, val, _ in self.alarm_log}
        alarm_values = [alarm_lookup[i] for i in alarm_points if i in alarm_lookup]

        # نقاط الألارم الخاصة بـ MR-chart
        mr_alarm_indices = []
        mr_alarm_values = []
        if self.ucl_mr is not None and mr:
            for j, mr_val in enumerate(mr, start=1):
                if mr_val > self.ucl_mr:
                    mr_alarm_indices.append(indices[j])
                    mr_alarm_values.append(mr_val)

        return {
            "indices": indices,
            "values": values,
            "mr": mr,
            "ucl_i": self.ucl_i,
            "lcl_i": self.lcl_i,
            "ucl_mr": self.ucl_mr,
            "target": self.target,
            "sigma_hat": self._current_sigma(),
            "mean": self.mean,
            "mr_bar": self.mr_bar,
            "alarms": list(self.alarm_log),
            "alarm_indices": alarm_points,
            "alarm_values": alarm_values,
            "mr_alarm_indices": mr_alarm_indices,
            "mr_alarm_values": mr_alarm_values,
            "comfort_low": self.comfort_low,
            "comfort_high": self.comfort_high,
            "calibrated": self.calibrated,
            "phase1_count": len(self.phase1_buffer),
            "phase1_size": self.phase1_size,
            "current": values[-1] if values else None,
        }