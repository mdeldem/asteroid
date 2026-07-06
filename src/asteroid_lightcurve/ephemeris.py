from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from .models import LightCurve
from .plotting import file_date_labels


HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
CORRECTION_VERSION = "geometry-pravec-cache-v2"


@dataclass
class FileEphemeris:
    file: str
    label: str
    observer: str
    object_command: str
    mid_jd: float
    ra_deg: float
    dec_deg: float
    r_au: float | None = None
    delta_au: float | None = None


@dataclass
class HjdCorrection:
    jd_utc: np.ndarray
    hjd_utc: np.ndarray
    correction_days: np.ndarray


@dataclass
class FileCorrection:
    jd_utc: np.ndarray
    hjd_utc: np.ndarray
    light_time_correction_days: np.ndarray
    mag_obs: np.ndarray
    mag_error: np.ndarray
    r_au: np.ndarray
    delta_au: np.ndarray
    geometry_correction: np.ndarray
    mag_reduced: np.ndarray
    ephemeris: FileEphemeris


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


def parse_horizons_geometry(result: str) -> list[tuple[float, float, float, float, float]]:
    try:
        table = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    except IndexError as exc:
        raise ValueError("Reponse Horizons sans table $$SOE/$$EOE") from exc

    rows: list[tuple[float, float, float, float, float]] = []
    for line in table.splitlines():
        fields = [field.strip() for field in line.split(",")]
        numeric: list[float] = []
        for field in fields:
            try:
                numeric.append(float(field))
            except ValueError:
                continue
        if len(numeric) < 6:
            continue
        offset = 1 if numeric[0] > 1_000_000.0 else 0
        if len(numeric) < offset + 6:
            continue
        jd = numeric[0] if offset else float("nan")
        ra_deg = numeric[offset]
        dec_deg = numeric[offset + 1]
        r_au = numeric[offset + 2]
        delta_au = numeric[offset + 4]
        rows.append((jd, ra_deg, dec_deg, r_au, delta_au))
    if not rows:
        raise ValueError("Geometrie r/delta introuvable dans la reponse Horizons")
    return rows


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


def fetch_horizons_geometry(
    object_command: str,
    jd_values: np.ndarray,
    timeout: float = 30.0,
) -> list[tuple[float, float, float, float, float]]:
    params = {
        "format": "json",
        "COMMAND": object_command,
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "500@399",
        "TLIST": "\n".join(f"{float(jd):.8f}" for jd in jd_values),
        "TLIST_TYPE": "JD",
        "TIME_TYPE": "UT",
        "QUANTITIES": "'1,19,20'",
        "CSV_FORMAT": "YES",
        "ANG_FORMAT": "DEG",
        "EXTRA_PREC": "YES",
    }
    url = f"{HORIZONS_API_URL}?{urlencode(params)}"
    with urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(f"Erreur Horizons: {payload['error']}")
    rows = parse_horizons_geometry(payload.get("result", ""))
    if all(np.isnan(row[0]) for row in rows):
        return [(float(jd), row[1], row[2], row[3], row[4]) for jd, row in zip(jd_values, rows)]
    return rows


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


def cache_paths(source: Path) -> tuple[Path, Path]:
    resolved = source.resolve()
    short_hash = sha1(str(resolved).lower().encode("utf-8")).hexdigest()[:10]
    cache_dir = resolved.parent / "cache"
    stem = f"{resolved.stem}.{short_hash}.correction-cache"
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.json"


def cache_metadata(source: Path, object_command: str) -> dict[str, object]:
    stat = source.stat()
    return {
        "source_file": source.name,
        "source_absolute_path": str(source.resolve()),
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
        "object_command": object_command,
        "observer_center": "500@399",
        "correction_version": CORRECTION_VERSION,
    }


def is_cache_valid(meta: dict[str, object], source: Path, object_command: str) -> bool:
    expected = cache_metadata(source, object_command)
    for key in ["source_absolute_path", "source_mtime_ns", "source_size", "object_command", "correction_version"]:
        if meta.get(key) != expected[key]:
            return False
    return source.exists()


def load_file_correction_from_cache(
    source: Path,
    object_command: str,
    label: str,
    observer: str,
) -> FileCorrection | None:
    csv_path, json_path = cache_paths(source)
    if not csv_path.exists() or not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not is_cache_valid(meta, source, object_command):
        return None

    data = np.genfromtxt(csv_path, delimiter=";", names=True, encoding="utf-8-sig")
    if data.size == 0:
        return None
    data = np.atleast_1d(data)
    forbidden = {"zi", "block_offset", "mag_plot", "residuals", "residual", "mag_aligned"}
    if forbidden.intersection({name.lower() for name in data.dtype.names or ()}):
        return None
    ephemeris = FileEphemeris(
        file=source.name,
        label=label,
        observer=observer,
        object_command=object_command,
        mid_jd=float(np.mean(data["jd_utc"])),
        ra_deg=float(meta.get("ra_deg", np.nan)),
        dec_deg=float(meta.get("dec_deg", np.nan)),
        r_au=float(np.mean(data["r_au"])),
        delta_au=float(np.mean(data["delta_au"])),
    )
    return FileCorrection(
        jd_utc=np.asarray(data["jd_utc"], dtype=float),
        hjd_utc=np.asarray(data["hjd_utc"], dtype=float),
        light_time_correction_days=np.asarray(data["light_time_correction_days"], dtype=float),
        mag_obs=np.asarray(data["mag_obs"], dtype=float),
        mag_error=np.asarray(data["mag_error"], dtype=float),
        r_au=np.asarray(data["r_au"], dtype=float),
        delta_au=np.asarray(data["delta_au"], dtype=float),
        geometry_correction=np.asarray(data["geometry_correction"], dtype=float),
        mag_reduced=np.asarray(data["mag_reduced"], dtype=float),
        ephemeris=ephemeris,
    )


def write_file_correction_cache(source: Path, object_command: str, correction: FileCorrection) -> None:
    csv_path, json_path = cache_paths(source)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "jd_utc",
        "hjd_utc",
        "light_time_correction_days",
        "mag_obs",
        "mag_error",
        "r_au",
        "delta_au",
        "geometry_correction",
        "mag_reduced",
    ]
    rows = np.column_stack(
        [
            correction.jd_utc,
            correction.hjd_utc,
            correction.light_time_correction_days,
            correction.mag_obs,
            correction.mag_error,
            correction.r_au,
            correction.delta_au,
            correction.geometry_correction,
            correction.mag_reduced,
        ]
    )
    np.savetxt(
        csv_path,
        rows,
        delimiter=";",
        header=";".join(header),
        comments="",
        fmt="%.12g",
        encoding="utf-8",
    )
    meta = cache_metadata(source, object_command)
    meta.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "software_version": None,
            "ra_deg": correction.ephemeris.ra_deg,
            "dec_deg": correction.ephemeris.dec_deg,
        }
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_file_correction(
    curve: LightCurve,
    group_id: int,
    object_command: str,
    label: str,
    timeout: float = 30.0,
    geometric_correction: bool = True,
) -> FileCorrection:
    obs = curve.files[group_id]
    mask = curve.group == group_id
    jd_utc = curve.jd[mask].copy()
    mag_obs = curve.mag_observed[mask].copy() if curve.mag_observed is not None else curve.magnitude[mask].copy()
    jd_grid = np.unique(np.asarray([float(np.min(jd_utc)), float(np.max(jd_utc))], dtype=float))
    rows = fetch_horizons_geometry(object_command, jd_grid, timeout=timeout)
    eph_jd = np.asarray([row[0] for row in rows], dtype=float)
    ra_values = np.asarray([row[1] for row in rows], dtype=float)
    dec_values = np.asarray([row[2] for row in rows], dtype=float)
    r_values = np.asarray([row[3] for row in rows], dtype=float)
    delta_values = np.asarray([row[4] for row in rows], dtype=float)
    if eph_jd.size == 1:
        r_au = np.full_like(jd_utc, r_values[0], dtype=float)
        delta_au = np.full_like(jd_utc, delta_values[0], dtype=float)
    else:
        order = np.argsort(eph_jd)
        r_au = np.interp(jd_utc, eph_jd[order], r_values[order])
        delta_au = np.interp(jd_utc, eph_jd[order], delta_values[order])

    ra_deg = float(np.interp(float(np.mean(jd_utc)), eph_jd, ra_values)) if eph_jd.size > 1 else float(ra_values[0])
    dec_deg = float(np.interp(float(np.mean(jd_utc)), eph_jd, dec_values)) if eph_jd.size > 1 else float(dec_values[0])
    light_time = heliocentric_correction_days(jd_utc, ra_deg, dec_deg)
    geometry = 5.0 * np.log10(r_au * delta_au) if geometric_correction else np.zeros_like(jd_utc)
    mag_reduced = mag_obs - geometry
    ephemeris = FileEphemeris(
        file=obs.path.name,
        label=label,
        observer=obs.observer_name,
        object_command=object_command,
        mid_jd=float(np.mean(jd_utc)),
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        r_au=float(np.mean(r_au)),
        delta_au=float(np.mean(delta_au)),
    )
    return FileCorrection(
        jd_utc=jd_utc,
        hjd_utc=jd_utc + light_time,
        light_time_correction_days=light_time,
        mag_obs=mag_obs,
        mag_error=curve.mag_error[mask].copy(),
        r_au=r_au,
        delta_au=delta_au,
        geometry_correction=geometry,
        mag_reduced=mag_reduced,
        ephemeris=ephemeris,
    )


def apply_ephemeris_corrections(
    curve: LightCurve,
    timeout: float = 30.0,
    use_cache: bool = True,
    geometric_correction: bool = True,
) -> tuple[list[FileEphemeris], HjdCorrection]:
    command = asteroid_command(curve)
    labels = file_date_labels(curve)
    jd_utc = curve.jd.copy()
    hjd_utc = np.empty_like(curve.jd, dtype=float)
    light_time = np.empty_like(curve.jd, dtype=float)
    mag_observed = curve.mag_observed.copy() if curve.mag_observed is not None else curve.magnitude.copy()
    mag_reduced = np.empty_like(curve.magnitude, dtype=float)
    r_au = np.empty_like(curve.magnitude, dtype=float)
    delta_au = np.empty_like(curve.magnitude, dtype=float)
    geometry = np.empty_like(curve.magnitude, dtype=float)
    ephemerides: list[FileEphemeris] = []

    for group_id, label in enumerate(labels):
        obs = curve.files[group_id]
        correction = load_file_correction_from_cache(obs.path, command, label, obs.observer_name) if use_cache else None
        if correction is None:
            correction = build_file_correction(
                curve,
                group_id,
                command,
                label,
                timeout=timeout,
                geometric_correction=geometric_correction,
            )
            if use_cache and geometric_correction:
                write_file_correction_cache(obs.path, command, correction)
        mask = curve.group == group_id
        if int(np.sum(mask)) != correction.jd_utc.size:
            raise ValueError(f"Cache incoherent pour {obs.path.name}: nombre de mesures different")
        hjd_utc[mask] = correction.hjd_utc
        light_time[mask] = correction.light_time_correction_days
        mag_observed[mask] = correction.mag_obs
        mag_reduced[mask] = correction.mag_reduced if geometric_correction else correction.mag_obs
        r_au[mask] = correction.r_au
        delta_au[mask] = correction.delta_au
        geometry[mask] = correction.geometry_correction if geometric_correction else 0.0
        ephemerides.append(correction.ephemeris)

    curve.jd_utc = jd_utc
    curve.jd = hjd_utc
    curve.time_label = "HJD"
    curve.light_time_correction_days = light_time
    curve.mag_observed = mag_observed
    curve.mag_reduced = mag_reduced
    curve.geometry_correction = geometry
    curve.r_au = r_au
    curve.delta_au = delta_au
    # After geometric correction, magnitude is the reduced magnitude used by the Pravec fit.
    curve.magnitude = mag_reduced
    return ephemerides, HjdCorrection(jd_utc=jd_utc, hjd_utc=hjd_utc, correction_days=light_time)


def clear_correction_cache(data_dir: Path, dry_run: bool = False) -> int:
    cache_dir = data_dir / "cache"
    if not cache_dir.exists():
        return 0
    paths = [path for path in cache_dir.iterdir() if path.is_file()]
    count = len(paths)
    if dry_run:
        return count
    for path in paths:
        path.unlink()
    try:
        cache_dir.rmdir()
    except OSError:
        pass
    return count


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
