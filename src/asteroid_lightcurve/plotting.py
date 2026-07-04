from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models import LightCurve
from .period import FitResult


def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def phase(jd: np.ndarray, period_hours: float) -> np.ndarray:
    period_days = period_hours / 24.0
    return ((jd - np.min(jd)) / period_days) % 1.0


def jd_to_date_label(jd: float) -> str:
    unix_epoch_jd = 2440587.5
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=jd - unix_epoch_jd)
    return dt.strftime("%Y-%m-%d")


def group_offsets(curve: LightCurve, fit: FitResult) -> np.ndarray:
    offsets = np.zeros(len(curve.group_names), dtype=float)
    for group_id in range(1, len(curve.group_names)):
        offsets[group_id] = fit.coefficients[group_id]
    return offsets


def corrected_magnitudes(curve: LightCurve, fit: FitResult) -> np.ndarray:
    offsets = group_offsets(curve, fit)
    return curve.magnitude - offsets[curve.group]


def corrected_model(curve: LightCurve, fit: FitResult) -> np.ndarray:
    offsets = group_offsets(curve, fit)
    return fit.model - offsets[curve.group]


def file_date_labels(curve: LightCurve) -> list[str]:
    dates: list[str] = []
    for group_id in range(len(curve.group_names)):
        group_jd = curve.jd[curve.group == group_id]
        dates.append(jd_to_date_label(float(np.min(group_jd))))

    counts = Counter(dates)
    seen: Counter[str] = Counter()
    labels: list[str] = []
    for date in dates:
        if counts[date] == 1:
            labels.append(date)
        else:
            seen[date] += 1
            labels.append(f"{date}-{seen[date]}")
    return labels


def plot_periodogram(
    periods_hours: np.ndarray,
    score: np.ndarray,
    best_period: float,
    ylabel: str,
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(periods_hours, score, color="tab:blue", lw=1.1)
    ax.axvline(best_period, color="tab:red", lw=1.0, ls="--", label=f"{best_period:.6f} h")
    ax.set_xlabel("Periode (h)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def model_at_phase(curve: LightCurve, fit: FitResult, model_phase: np.ndarray) -> np.ndarray:
    model_angle = 2.0 * np.pi * model_phase
    model = np.full_like(model_phase, fit.coefficients[0])
    harmonic_start = 1 + max(len(curve.group_names) - 1, 0)
    for harmonic in range(1, fit.order + 1):
        cos_coef = fit.coefficients[harmonic_start + 2 * (harmonic - 1)]
        sin_coef = fit.coefficients[harmonic_start + 2 * (harmonic - 1) + 1]
        model += cos_coef * np.cos(harmonic * model_angle)
        model += sin_coef * np.sin(harmonic * model_angle)
    return model


def plot_folded_lightcurve(
    curve: LightCurve,
    fit: FitResult,
    path: str | Path,
    by_file: bool = False,
) -> None:
    ph = phase(curve.jd, fit.period_hours)
    doubled_phase = np.concatenate([ph, ph + 1.0])
    mag = corrected_magnitudes(curve, fit)
    doubled_mag = np.concatenate([mag, mag])
    doubled_err = np.concatenate([curve.mag_error, curve.mag_error])

    model_phase = np.linspace(0.0, 2.0, 600)
    model = model_at_phase(curve, fit, model_phase)

    fig, ax = plt.subplots(figsize=(9, 5))
    if by_file:
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
        labels = file_date_labels(curve)
        for group_id, label in enumerate(labels):
            mask = curve.group == group_id
            group_phase = np.concatenate([ph[mask], ph[mask] + 1.0])
            group_mag = np.concatenate([mag[mask], mag[mask]])
            group_err = np.concatenate([curve.mag_error[mask], curve.mag_error[mask]])
            ax.errorbar(
                group_phase,
                group_mag,
                yerr=group_err,
                fmt=markers[group_id % len(markers)],
                ms=4,
                alpha=0.78,
                capsize=0,
                elinewidth=0.7,
                label=label,
            )
        ax.legend(ncol=2, fontsize="small")
    else:
        ax.errorbar(
            doubled_phase,
            doubled_mag,
            yerr=doubled_err,
            fmt=".",
            ms=4,
            alpha=0.75,
            ecolor="0.7",
            color="tab:blue",
        )
    ax.plot(model_phase, model, color="tab:red", lw=1.5)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Magnitude corrigee")
    ax.set_title(f"Courbe repliee - P = {fit.period_hours:.6f} h, ordre {fit.order}")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_residuals(curve: LightCurve, fit: FitResult, path: str | Path) -> None:
    ph = phase(curve.jd, fit.period_hours)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    axes[0].errorbar(curve.jd, fit.residuals, yerr=curve.mag_error, fmt=".", color="tab:purple")
    axes[0].axhline(0.0, color="0.2", lw=1.0)
    axes[0].set_xlabel("JD")
    axes[0].set_ylabel("Residus (mag)")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(ph, fit.residuals, yerr=curve.mag_error, fmt=".", color="tab:green")
    axes[1].axhline(0.0, color="0.2", lw=1.0)
    axes[1].set_xlabel("Phase")
    axes[1].set_ylabel("Residus (mag)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
