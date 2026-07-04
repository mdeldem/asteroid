from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .models import LightCurve
from .period import FitResult


def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def phase(jd: np.ndarray, period_days: float) -> np.ndarray:
    return ((jd - np.min(jd)) / period_days) % 1.0


def format_period(period_days: float) -> str:
    return f"{period_days * 24.0:.6f} h ({period_days:.8f} j)"


def format_period_with_uncertainty(period_days: float, uncertainty_days: float | None) -> str:
    if uncertainty_days is None:
        return format_period(period_days)
    return (
        f"{period_days * 24.0:.6f} +/- {uncertainty_days * 24.0:.6f} h "
        f"({period_days:.8f} +/- {uncertainty_days:.8f} j)"
    )


def jd_to_date_label(jd: float) -> str:
    unix_epoch_jd = 2440587.5
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=jd - unix_epoch_jd)
    return dt.strftime("%Y-%m-%d")


def format_t0(curve: LightCurve) -> str:
    t0_index = int(np.argmin(curve.jd))
    decimals = int(curve.jd_decimals[t0_index]) if curve.jd_decimals.size else 8
    return f"{float(curve.jd[t0_index]):.{decimals}f}"


def object_title(curve: LightCurve) -> str:
    for obs in curve.files:
        if obs.object_name:
            return obs.object_name
    return "Objet"


def group_offsets(curve: LightCurve, fit: FitResult) -> np.ndarray:
    offsets = np.zeros(len(curve.group_names), dtype=float)
    for group_id in range(1, len(curve.group_names)):
        offsets[group_id] = fit.coefficients[group_id]
    return offsets


def aligned_magnitudes(curve: LightCurve, fit: FitResult) -> np.ndarray:
    offsets = group_offsets(curve, fit)
    return curve.magnitude - offsets[curve.group]


def aligned_model(curve: LightCurve, fit: FitResult) -> np.ndarray:
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
            labels.append(f"[{date}]")
        else:
            seen[date] += 1
            labels.append(f"[{date}-{seen[date]}]")
    return labels


def model_amplitude(fit: FitResult) -> float:
    model_phase = np.linspace(0.0, 1.0, 1000)
    model = model_at_phase_for_coefficients(fit, model_phase)
    return float(np.nanmax(model) - np.nanmin(model))


def per_file_scatter(curve: LightCurve, fit: FitResult) -> np.ndarray:
    aligned_residuals = aligned_magnitudes(curve, fit) - aligned_model(curve, fit)
    scatters = np.full(len(curve.group_names), np.nan, dtype=float)
    for group_id in range(len(curve.group_names)):
        values = aligned_residuals[curve.group == group_id]
        if values.size >= 2:
            scatters[group_id] = float(np.std(values, ddof=1))
        elif values.size == 1:
            scatters[group_id] = 0.0
    return scatters


def file_plot_styles(count: int) -> tuple[list[str], list[str]]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
    return colors, markers[:count]


def file_legend_labels(curve: LightCurve, fit: FitResult) -> list[str]:
    date_labels = file_date_labels(curve)
    scatters = per_file_scatter(curve, fit)
    labels: list[str] = []
    for group_id, date_label in enumerate(date_labels):
        observer = curve.files[group_id].observer_name or "?"
        if len(observer) > 30:
            observer = observer[:30]
        scatter = scatters[group_id]
        scatter_label = "σ=?" if np.isnan(scatter) else f"σ={scatter * 1000.0:.0f} mmag"
        labels.append(f"{date_label} {observer} ({scatter_label})")
    return labels


def add_file_legend(ax: plt.Axes, curve: LightCurve, fit: FitResult, uniform_style: bool = False) -> None:
    colors, markers = file_plot_styles(len(curve.group_names))
    labels = file_legend_labels(curve, fit)
    handles = [
        Line2D(
            [0],
            [0],
            marker="." if uniform_style else markers[group_id % len(markers)],
            color="tab:blue" if uniform_style else colors[group_id % len(colors)],
            linestyle="None",
            markersize=7 if uniform_style else 5,
            alpha=0.75 if uniform_style else 0.7,
        )
        for group_id in range(len(labels))
    ]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize="small",
        frameon=True,
        borderaxespad=0.0,
    )


def plot_periodogram(
    periods_days: np.ndarray,
    score: np.ndarray,
    best_period: float,
    ylabel: str,
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(periods_days, score, color="tab:blue", lw=1.1)
    ax.axvline(best_period, color="tab:red", lw=1.0, ls="--", label=format_period(best_period))
    ax.set_xlabel("Periode (j)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def model_at_phase_for_coefficients(fit: FitResult, model_phase: np.ndarray) -> np.ndarray:
    model_angle = 2.0 * np.pi * model_phase
    model = np.full_like(model_phase, fit.coefficients[0])
    harmonic_start = len(fit.coefficients) - 2 * fit.order
    for harmonic in range(1, fit.order + 1):
        cos_coef = fit.coefficients[harmonic_start + 2 * (harmonic - 1)]
        sin_coef = fit.coefficients[harmonic_start + 2 * (harmonic - 1) + 1]
        model += cos_coef * np.cos(harmonic * model_angle)
        model += sin_coef * np.sin(harmonic * model_angle)
    return model


def model_at_phase(curve: LightCurve, fit: FitResult, model_phase: np.ndarray) -> np.ndarray:
    return model_at_phase_for_coefficients(fit, model_phase)


def folded_axis_limits(curve: LightCurve, fit: FitResult) -> tuple[tuple[float, float], tuple[float, float]]:
    mag = aligned_magnitudes(curve, fit)
    model_phase = np.linspace(0.0, 2.0, 600)
    model = model_at_phase(curve, fit, model_phase)
    y_values = np.concatenate([mag - curve.mag_error, mag + curve.mag_error, model])
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    padding = max((y_max - y_min) * 0.05, 0.01)
    return (-0.1, 2.1), (y_max + padding, y_min - padding)


def add_folded_title(
    ax: plt.Axes,
    curve: LightCurve,
    fit: FitResult,
    period_uncertainty_days: float | None,
) -> None:
    ax.text(
        0.0,
        1.145,
        object_title(curve),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=17,
    )
    ax.text(
        0.0,
        1.075,
        f"P = {format_period_with_uncertainty(fit.period_days, period_uncertainty_days)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
    )
    ax.text(
        0.0,
        1.030,
        f"ordre {fit.order}, T0 = {format_t0(curve)} JD, A = {model_amplitude(fit):.3f} mag",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
    )


def plot_file_groups(
    ax: plt.Axes,
    curve: LightCurve,
    y_values: np.ndarray,
    y_errors: np.ndarray,
    fit: FitResult,
    alpha: float = 0.42,
) -> None:
    ph = phase(curve.jd, fit.period_days)
    colors, markers = file_plot_styles(len(curve.group_names))
    labels = file_legend_labels(curve, fit)
    for group_id, label in enumerate(labels):
        mask = curve.group == group_id
        group_phase = np.concatenate([ph[mask], ph[mask] + 1.0])
        group_y = np.concatenate([y_values[mask], y_values[mask]])
        group_err = np.concatenate([y_errors[mask], y_errors[mask]])
        ax.errorbar(
            group_phase,
            group_y,
            yerr=group_err,
            fmt=markers[group_id % len(markers)],
            color=colors[group_id % len(colors)],
            ms=3.5,
            alpha=alpha,
            capsize=0,
            elinewidth=0.45,
            label=label,
        )


def plot_folded_lightcurve(
    curve: LightCurve,
    fit: FitResult,
    path: str | Path,
    by_file: bool = False,
    period_uncertainty_days: float | None = None,
) -> None:
    ph = phase(curve.jd, fit.period_days)
    doubled_phase = np.concatenate([ph, ph + 1.0])
    mag = aligned_magnitudes(curve, fit)
    doubled_mag = np.concatenate([mag, mag])
    doubled_err = np.concatenate([curve.mag_error, curve.mag_error])

    model_phase = np.linspace(0.0, 2.0, 600)
    model = model_at_phase(curve, fit, model_phase)

    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    if by_file:
        plot_file_groups(ax, curve, mag, curve.mag_error, fit)
        add_file_legend(ax, curve, fit)
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
        add_file_legend(ax, curve, fit, uniform_style=True)
    ax.plot(model_phase, model, color="tab:red", lw=1.5)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Magnitude alignee")
    add_folded_title(ax, curve, fit, period_uncertainty_days)
    xlim, ylim = folded_axis_limits(curve, fit)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.88))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_folded_lightcurve_by_file_with_residuals(
    curve: LightCurve,
    fit: FitResult,
    path: str | Path,
    period_uncertainty_days: float | None = None,
) -> None:
    mag = aligned_magnitudes(curve, fit)
    residuals = mag - aligned_model(curve, fit)
    model_phase = np.linspace(0.0, 2.0, 600)
    model = model_at_phase(curve, fit, model_phase)

    fig, (ax_lc, ax_res) = plt.subplots(
        2,
        1,
        figsize=(13.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.08},
    )

    plot_file_groups(ax_lc, curve, mag, curve.mag_error, fit)
    ax_lc.plot(model_phase, model, color="tab:red", lw=1.5)
    add_file_legend(ax_lc, curve, fit)
    add_folded_title(ax_lc, curve, fit, period_uncertainty_days)
    xlim, ylim = folded_axis_limits(curve, fit)
    ax_lc.set_xlim(*xlim)
    ax_lc.set_ylim(*ylim)
    ax_lc.set_ylabel("Magnitude alignee")
    ax_lc.grid(alpha=0.25)
    ax_lc.tick_params(labelbottom=False)

    plot_file_groups(ax_res, curve, residuals, curve.mag_error, fit, alpha=0.55)
    ax_res.axhline(0.0, color="0.2", lw=1.0)
    y_values = np.concatenate([residuals - curve.mag_error, residuals + curve.mag_error])
    max_abs = max(float(np.nanmax(np.abs(y_values))), 0.01)
    ax_res.set_ylim(-max_abs * 1.15, max_abs * 1.15)
    ax_res.set_xlim(*xlim)
    ax_res.set_xlabel("Phase")
    ax_res.set_ylabel("Residus (mag)")
    ax_res.grid(alpha=0.25)

    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.09, top=0.82, hspace=0.08)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(curve: LightCurve, fit: FitResult, path: str | Path) -> None:
    ph = phase(curve.jd, fit.period_days)
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
