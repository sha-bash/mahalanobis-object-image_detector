from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass
class StageTimer:
    enabled: bool = True
    values_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.values_ms[stage].append((time.perf_counter() - started) * 1000.0)

    def add(self, stage: str, elapsed_ms: float) -> None:
        if self.enabled:
            self.values_ms[stage].append(float(elapsed_ms))

    def merge(self, other: StageTimer | None) -> None:
        if other is None:
            return
        for stage, values in other.values_ms.items():
            self.values_ms[stage].extend(values)

    def summary(self) -> dict:
        stages = {}
        for name, raw_values in sorted(self.values_ms.items()):
            values = np.asarray(raw_values, dtype=float)
            stages[name] = {
                "count": int(values.size),
                "total_ms": float(values.sum()),
                "mean_ms": float(values.mean()),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
            }
        frame_totals = self.values_ms.get("frame_total", [])
        total_ms = float(sum(frame_totals))
        return {
            "stages": stages,
            "total_inference_ms": total_ms,
            "mean_frame_ms": total_ms / len(frame_totals) if frame_totals else 0.0,
            "throughput_fps": 1000.0 * len(frame_totals) / total_ms if total_ms > 0 else 0.0,
        }
