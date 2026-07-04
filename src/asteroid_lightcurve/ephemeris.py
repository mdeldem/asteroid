from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from .models import LightCurve
from .plotting import file_date_labels


HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"


@dataclass
class FileEphemeris:
    file: str
    label: str
    observer: str
    object_command: str
    mid_jd: float
    ra_deg: float
    dec_deg: float


@dataclass
class HjdCorrection:
    jd_utc: np.ndarray
    hjd_utc: np.ndarray
    correction_days: np.ndarray


def asteroid_command(curve: LightCurve) -> str:
    for obs in curve.files:
        match = re.search(r"\((\d+)\)", obs.object_name)
        if match:
            return f"{match.group(1)};"
    raise ValueError("Impossible de trouver le numero d'asteroide dans le champ NOM")


def file_mid_jd(curve: LightCurve, group_id: int) -> float:
    values = curve.jd[curve.group == group_id]
    if values.size == 0:
        raise ValueError(f"Aucune mesure pour le groupe {group_id}")
    return float((np.min(values) + np.max(values)) / 2.0)


def parse_horizons_ra_dec(result: str) -> tuple[float, float]:
    try:
        table = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    except IndexError as exc:
        raise ValueError("Reponse Horizons sans table $$SOE/$$EOE") from exc

    for line in table.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        numeric = []
        for field in fields:
            try:
                numeric.append(float(field))
            except ValueError:
                continue
        if len(numeric) >= 2:
            return numeric[0], numeric[1]
    raise ValueError("RA/DEC introuvables dans la reponse Horizons")


def fetch_horizons_ra_dec(object_command: str, jd: float, timeout: float = 30.0) -> tuple[float, float]:
    params = {
        "format": "json",
        "COMMAND": object_command,
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "500@399",
        "TLIST": f"{jd:.8f}",
        "TLIST_TYPE": "JD",
        "TIME_TYPE": "UT",
        "QUANTITIES": "1",
        "CSV_FORMAT": "YES",
        "ANG_FORMAT": "DEG",
        "EXTRA_PREC": "YES",
    }
    url = f"{HORIZONS_API_URL}?{urlencode(params)}"
    with urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(f"Erreur Horizons: {payload['error']}")
    return parse_horizons_ra_dec(payload.get("result", ""))


def fetch_file_ephemerides(curve: LightCurve, timeout: float = 30.0) -> list[FileEphemeris]:
    command = asteroid_command(curve)
    labels = file_date_labels(curve)
    ephemerides: list[FileEphemeris] = []
    for group_id, label in enumerate(labels):
        mid_jd = file_mid_jd(curve, group_id)
        ra_deg, dec_deg = fetch_horizons_ra_dec(command, mid_jd, timeout=timeout)
        ephemerides.append(
            FileEphemeris(
                file=curve.group_names[group_id],
                label=label,
                observer=curve.files[group_id].observer_name,
                object_command=command,
                mid_jd=mid_jd,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
            )
        )
    return ephemerides


def heliocentric_correction_days(jd_utc: np.ndarray, ra_deg: float, dec_deg: float) -> np.ndarray:
    try:
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time
        import astropy.units as u
    except ImportError as exc:
        raise RuntimeError(
            "Le calcul HJD necessite astropy. Installe les dependances avec `uv sync`."
        ) from exc

    geocenter = EarthLocation.from_geocentric(0.0 * u.m, 0.0 * u.m, 0.0 * u.m)
    times = Time(jd_utc, format="jd", scale="utc", location=geocenter)
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    correction = times.light_travel_time(target, kind="heliocentric")
    return correction.to_value(u.day)


def apply_hjd_correction(curve: LightCurve, ephemerides: list[FileEphemeris]) -> HjdCorrection:
    if len(ephemerides) != len(curve.group_names):
        raise ValueError("Le nombre d'ephemerides ne correspond pas au nombre de fichiers")

    jd_utc = curve.jd.copy()
    correction_days = np.zeros_like(curve.jd, dtype=float)
    for group_id, eph in enumerate(ephemerides):
        mask = curve.group == group_id
        correction_days[mask] = heliocentric_correction_days(curve.jd[mask], eph.ra_deg, eph.dec_deg)

    hjd_utc = jd_utc + correction_days
    curve.jd = hjd_utc
    curve.time_label = "HJD"
    return HjdCorrection(jd_utc=jd_utc, hjd_utc=hjd_utc, correction_days=correction_days)
