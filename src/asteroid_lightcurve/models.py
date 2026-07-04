from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ObservationFile:
    path: Path
    fmt: str
    keywords: dict[str, list[str]] = field(default_factory=dict)
    jd: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    jd_decimals: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    magnitude: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    mag_error: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    ignored: list[list[str]] = field(default_factory=list)

    @property
    def object_name(self) -> str:
        return " ".join(self.keywords.get("NOM", []))

    @property
    def observer(self) -> str:
        return " ".join(self.keywords.get("MES", []))

    @property
    def observer_name(self) -> str:
        return self.observer.split("@", 1)[0].strip()

    @property
    def filter_name(self) -> str:
        values = self.keywords.get("FIL", [])
        return values[0] if values else ""

    @property
    def exposure_seconds(self) -> float:
        values = self.keywords.get("POS", [])
        if len(values) >= 2:
            try:
                return float(values[1])
            except ValueError:
                return 0.0
        return 0.0

    @property
    def exposure_time_position(self) -> int:
        values = self.keywords.get("POS", [])
        if values:
            try:
                position = int(values[0])
            except ValueError:
                return 0
            if position in {-1, 0, 1}:
                return position
        return 0

    def mid_exposure_jd(self) -> np.ndarray:
        if self.exposure_seconds > 0:
            half_exposure_days = self.exposure_seconds / 2.0 / 86400.0
            if self.exposure_time_position == -1:
                return self.jd + half_exposure_days
            if self.exposure_time_position == 1:
                return self.jd - half_exposure_days
        return self.jd.copy()


@dataclass
class LightCurve:
    jd: np.ndarray
    jd_decimals: np.ndarray
    magnitude: np.ndarray
    mag_error: np.ndarray
    group: np.ndarray
    group_names: list[str]
    files: list[ObservationFile]

    @property
    def n_points(self) -> int:
        return int(self.jd.size)

    @property
    def baseline_days(self) -> float:
        if self.jd.size < 2:
            return 0.0
        return float(np.max(self.jd) - np.min(self.jd))
