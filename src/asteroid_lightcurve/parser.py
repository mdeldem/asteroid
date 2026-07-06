from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import LightCurve, ObservationFile


KEYWORDS = {"FMT", "NOM", "MES", "POS", "FIL", "CAT", "TEL", "CAP"}


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob(pattern)]
        paths.extend(matches if matches else [Path(pattern)])
    unique = sorted({p.resolve() for p in paths})
    missing = [str(p) for p in unique if not p.exists()]
    if missing:
        raise FileNotFoundError("Fichier introuvable: " + ", ".join(missing))
    return unique


def read_observation_file(path: str | Path, encoding: str = "mbcs") -> ObservationFile:
    path = Path(path)
    fmt = ""
    keywords: dict[str, list[str]] = {}
    jd: list[float] = []
    jd_decimals: list[int] = []
    mag: list[float] = []
    err: list[float] = []
    ignored: list[list[str]] = []

    with path.open("r", encoding=encoding, errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue

            parts = line.split()
            key = parts[0]
            if key in KEYWORDS:
                values = parts[1:]
                keywords[key] = values
                if key == "FMT":
                    fmt = values[0] if values else ""
                continue

            if not fmt:
                raise ValueError(f"{path}:{line_number}: mesure trouvee avant FMT")
            if len(parts) < len(fmt):
                raise ValueError(
                    f"{path}:{line_number}: {len(parts)} champs pour FMT {fmt!r}"
                )

            current_ignored: list[str] = []
            row_jd = row_mag = row_err = None
            for marker, value in zip(fmt, parts):
                if marker == "D":
                    row_jd = float(value)
                    jd_decimals.append(len(value.split(".", 1)[1]) if "." in value else 0)
                elif marker == "V":
                    row_mag = float(value)
                elif marker == "v":
                    row_err = float(value)
                elif marker == "x":
                    current_ignored.append(value)

            if row_jd is None or row_mag is None or row_err is None:
                raise ValueError(f"{path}:{line_number}: FMT doit contenir D, V et v")
            jd.append(row_jd)
            mag.append(row_mag)
            err.append(row_err)
            ignored.append(current_ignored)

    if not fmt:
        raise ValueError(f"{path}: mot-cle FMT manquant")
    if not jd:
        raise ValueError(f"{path}: aucune mesure")

    return ObservationFile(
        path=path,
        fmt=fmt,
        keywords=keywords,
        jd=np.asarray(jd, dtype=float),
        jd_decimals=np.asarray(jd_decimals, dtype=int),
        magnitude=np.asarray(mag, dtype=float),
        mag_error=np.asarray(err, dtype=float),
        ignored=ignored,
    )


def read_lightcurve(paths: Iterable[str | Path], use_mid_exposure: bool = True) -> LightCurve:
    files = [read_observation_file(path) for path in paths]
    jd_parts: list[np.ndarray] = []
    mag_parts: list[np.ndarray] = []
    err_parts: list[np.ndarray] = []
    jd_decimal_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    group_names: list[str] = []

    for group_id, obs in enumerate(files):
        group_names.append(obs.path.name)
        jd = obs.mid_exposure_jd() if use_mid_exposure else obs.jd.copy()
        jd_parts.append(jd)
        jd_decimal_parts.append(obs.jd_decimals)
        mag_parts.append(obs.magnitude)
        err_parts.append(obs.mag_error)
        group_parts.append(np.full(obs.jd.shape, group_id, dtype=int))

    order = np.argsort(np.concatenate(jd_parts))
    magnitude = np.concatenate(mag_parts)[order]
    return LightCurve(
        jd=np.concatenate(jd_parts)[order],
        jd_decimals=np.concatenate(jd_decimal_parts)[order],
        magnitude=magnitude,
        mag_error=np.concatenate(err_parts)[order],
        group=np.concatenate(group_parts)[order],
        group_names=group_names,
        files=files,
        mag_observed=magnitude.copy(),
    )
