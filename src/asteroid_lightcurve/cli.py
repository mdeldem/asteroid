from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from .ephemeris import FileEphemeris, HjdCorrection, apply_hjd_correction, fetch_file_ephemerides
from .models import LightCurve, subset_lightcurve
from .parser import expand_inputs, read_lightcurve
from .period import (
    CandidateFit,
    PeriodSelection,
    PeriodUncertainty,
    estimate_period_uncertainty,
    gls_power,
    period_grid,
    refine_period,
    search_fourier,
    search_period_order_candidates,
)
from .plotting import (
    aligned_magnitudes,
    aligned_model,
    ensure_outdir,
    file_date_labels,
    model_amplitude,
    per_file_scatter,
    format_period,
    plot_folded_lightcurve_by_file_with_residuals,
    plot_folded_lightcurve,
    plot_periodogram,
    plot_residual_filter,
    plot_residuals,
)


@dataclass
class ResidualFilter:
    keep_mask: np.ndarray
    reject_mask: np.ndarray
    residual_center: float
    residual_scale: float
    threshold_mag: float
    requested_threshold_mag: float


def parse_orders(value: str) -> range:
    if ":" in value:
        start, end = value.split(":", 1)
        return range(int(start), int(end) + 1)
    order = int(value)
    return range(order, order + 1)


def parse_multipliers(value: str) -> tuple[float, ...]:
    multipliers = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not multipliers:
        raise argparse.ArgumentTypeError("La liste des multiplicateurs ne doit pas etre vide")
    if any(multiplier <= 0 for multiplier in multipliers):
        raise argparse.ArgumentTypeError("Les multiplicateurs doivent etre strictement positifs")
    return multipliers


def format_uncertainty(value_days: float | None) -> str:
    if value_days is None:
        return "non bornee"
    return f"{value_days * 24.0:.6f} h ({value_days:.8f} j)"


def write_period_summary(
    path: Path,
    best_period_days: float,
    amplitude_mag: float,
    raw_uncertainty: PeriodUncertainty,
    scaled_uncertainty: PeriodUncertainty,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "kind",
                "period_days",
                "period_hours",
                "amplitude_mag",
                "delta_chi2",
                "lower_days",
                "upper_days",
                "minus_days",
                "plus_days",
                "symmetric_days",
                "lower_hours",
                "upper_hours",
                "minus_hours",
                "plus_hours",
                "symmetric_hours",
            ]
        )
        for kind, uncertainty in [
            ("delta_chi2_1", raw_uncertainty),
            ("scaled_delta_chi2", scaled_uncertainty),
        ]:
            writer.writerow(
                [
                    kind,
                    f"{best_period_days:.10f}",
                    f"{best_period_days * 24.0:.10f}",
                    f"{amplitude_mag:.6f}",
                    f"{uncertainty.delta_chi2:.6f}",
                    "" if uncertainty.lower_days is None else f"{uncertainty.lower_days:.10f}",
                    "" if uncertainty.upper_days is None else f"{uncertainty.upper_days:.10f}",
                    "" if uncertainty.minus_days is None else f"{uncertainty.minus_days:.10f}",
                    "" if uncertainty.plus_days is None else f"{uncertainty.plus_days:.10f}",
                    "" if uncertainty.symmetric_days is None else f"{uncertainty.symmetric_days:.10f}",
                    "" if uncertainty.lower_hours is None else f"{uncertainty.lower_hours:.10f}",
                    "" if uncertainty.upper_hours is None else f"{uncertainty.upper_hours:.10f}",
                    "" if uncertainty.minus_hours is None else f"{uncertainty.minus_hours:.10f}",
                    "" if uncertainty.plus_hours is None else f"{uncertainty.plus_hours:.10f}",
                    "" if uncertainty.symmetric_hours is None else f"{uncertainty.symmetric_hours:.10f}",
                ]
            )


def write_file_summary(path: Path, curve, fit) -> None:
    scatters = per_file_scatter(curve, fit)
    labels = file_date_labels(curve)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "file",
                "label",
                "observer",
                "n_points",
                "scatter_mag",
                "offset_mag",
            ]
        )
        for group_id, label in enumerate(labels):
            mask = curve.group == group_id
            offset = 0.0 if group_id == 0 else fit.coefficients[group_id]
            writer.writerow(
                [
                    curve.group_names[group_id],
                    label,
                    curve.files[group_id].observer_name,
                    int(np.sum(mask)),
                    "" if np.isnan(scatters[group_id]) else f"{scatters[group_id]:.6f}",
                    f"{offset:.6f}",
                ]
            )


def write_candidate_summary(path: Path, candidate_fits: list[CandidateFit]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "rank_bic",
                "gls_rank",
                "gls_period_days",
                "gls_period_hours",
                "gls_power",
                "multiplier",
                "candidate_period_days",
                "candidate_period_hours",
                "order",
                "refined_period_days",
                "refined_period_hours",
                "chi2",
                "reduced_chi2",
                "aic",
                "aicc",
                "bic",
                "last_harmonic_significance",
            ]
        )
        for rank, candidate_fit in enumerate(candidate_fits, start=1):
            candidate = candidate_fit.candidate
            fit = candidate_fit.fit
            writer.writerow(
                [
                    rank,
                    candidate.rank,
                    f"{candidate.gls_period_days:.10f}",
                    f"{candidate.gls_period_hours:.10f}",
                    f"{candidate.gls_power:.8f}",
                    f"{candidate.multiplier:.6f}",
                    f"{candidate.period_days:.10f}",
                    f"{candidate.period_hours:.10f}",
                    fit.order,
                    f"{fit.period_days:.10f}",
                    f"{fit.period_hours:.10f}",
                    f"{fit.chi2:.6f}",
                    f"{fit.reduced_chi2:.6f}",
                    f"{fit.aic:.6f}",
                    f"{fit.aicc:.6f}",
                    f"{fit.bic:.6f}",
                    "" if not fit.harmonic_significance else f"{fit.harmonic_significance[-1]:.6f}",
                ]
            )


def write_order_summary(path: Path, order_bests) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "order",
                "period_days",
                "period_hours",
                "chi2",
                "reduced_chi2",
                "aic",
                "aicc",
                "bic",
                "last_harmonic_significance",
            ]
        )
        for fit in order_bests:
            writer.writerow(
                [
                    fit.order,
                    f"{fit.period_days:.10f}",
                    f"{fit.period_hours:.10f}",
                    f"{fit.chi2:.6f}",
                    f"{fit.reduced_chi2:.6f}",
                    f"{fit.aic:.6f}",
                    f"{fit.aicc:.6f}",
                    f"{fit.bic:.6f}",
                    "" if not fit.harmonic_significance else f"{fit.harmonic_significance[-1]:.6f}",
                ]
            )


def write_period_selection_summary(path: Path, selection: PeriodSelection) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "kind",
                "strategy",
                "order",
                "period_days",
                "period_hours",
                "chi2",
                "reduced_chi2",
                "aic",
                "aicc",
                "bic",
                "stability_tolerance",
                "harmonic_significance_threshold",
                "last_harmonic_significance",
            ]
        )
        rows = [
            ("selected", selection.selected_fit),
            ("reference_order", selection.reference_fit),
            ("global_bic_best", selection.bic_best_fit),
        ]
        rows.extend((f"stable_order_{fit.order}", fit) for fit in selection.stable_order_fits)
        for kind, fit in rows:
            writer.writerow(
                [
                    kind,
                    selection.strategy,
                    fit.order,
                    f"{fit.period_days:.10f}",
                    f"{fit.period_hours:.10f}",
                    f"{fit.chi2:.6f}",
                    f"{fit.reduced_chi2:.6f}",
                    f"{fit.aic:.6f}",
                    f"{fit.aicc:.6f}",
                    f"{fit.bic:.6f}",
                    f"{selection.stability_tolerance:.6f}",
                    f"{selection.harmonic_significance_threshold:.6f}",
                    "" if not fit.harmonic_significance else f"{fit.harmonic_significance[-1]:.6f}",
                ]
            )


def package_version() -> str | None:
    try:
        return metadata.version("asteroid-lightcurve")
    except metadata.PackageNotFoundError:
        return None


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    input_paths: list[Path],
    curve: LightCurve,
    produced_files: list[str],
    best,
    residual_best,
    selection: PeriodSelection | None,
) -> None:
    parameters = {
        "files": list(args.files),
        "min_period_days": args.min_period,
        "max_period_days": args.max_period,
        "period_days": args.period,
        "samples": args.samples,
        "orders": args.orders,
        "gls_candidates": args.gls_candidates,
        "gls_multipliers": list(args.gls_multipliers),
        "refine_width": args.refine_width,
        "candidate_refine_samples": args.candidate_refine_samples,
        "refine_samples": args.refine_samples,
        "residual_filter": args.residual_filter,
        "residual_filter_sigma": args.residual_filter_sigma,
        "residual_filter_threshold_mag": args.residual_filter_threshold_mag,
        "residual_filter_max_reject_fraction": args.residual_filter_max_reject_fraction,
        "residual_filter_min_points": args.residual_filter_min_points,
        "keep_start_time": args.keep_start_time,
        "no_ephemeris": args.no_ephemeris,
        "horizons_timeout": args.horizons_timeout,
        "out": args.out,
    }
    data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": subprocess.list2cmdline(sys.argv),
        "argv": sys.argv,
        "package": {
            "name": "asteroid-lightcurve",
            "version": package_version(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "parameters": parameters,
        "inputs": {
            "patterns": list(args.files),
            "expanded_files": [str(input_path) for input_path in input_paths],
        },
        "data": {
            "n_files": len(curve.files),
            "n_points": int(curve.n_points),
            "baseline_days": float(curve.baseline_days),
        },
        "results": {
            "period_days": float(best.period_days),
            "period_hours": float(best.period_hours),
            "fourier_order": int(best.order),
            "reduced_chi2": float(best.reduced_chi2),
            "aic": float(best.aic),
            "aicc": float(best.aicc),
            "bic": float(best.bic),
            "residual_filtered_period_days": None if residual_best is None else float(residual_best.period_days),
            "residual_filtered_period_hours": None if residual_best is None else float(residual_best.period_hours),
            "residual_filtered_fourier_order": None if residual_best is None else int(residual_best.order),
            "residual_filtered_bic": None if residual_best is None else float(residual_best.bic),
        },
        "produced_files": produced_files,
    }
    if selection is not None:
        data["selection"] = {
            "strategy": selection.strategy,
            "stability_tolerance": float(selection.stability_tolerance),
            "reference_order": int(selection.reference_fit.order),
            "reference_period_days": float(selection.reference_fit.period_days),
            "selected_order": int(selection.selected_fit.order),
            "selected_period_days": float(selection.selected_fit.period_days),
            "global_bic_best_order": int(selection.bic_best_fit.order),
            "global_bic_best_period_days": float(selection.bic_best_fit.period_days),
            "global_bic_best_bic": float(selection.bic_best_fit.bic),
            "harmonic_significance_threshold": float(selection.harmonic_significance_threshold),
            "stable_orders": [
                {
                    "order": int(fit.order),
                    "period_days": float(fit.period_days),
                    "period_hours": float(fit.period_hours),
                    "reduced_chi2": float(fit.reduced_chi2),
                    "aicc": float(fit.aicc),
                    "bic": float(fit.bic),
                    "last_harmonic_significance": None
                    if not fit.harmonic_significance
                    else float(fit.harmonic_significance[-1]),
                }
                for fit in selection.stable_order_fits
            ],
        }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def residual_filter_from_fit(
    fit,
    sigma: float,
    threshold_mag: float | None,
    max_reject_fraction: float,
    min_points: int,
) -> ResidualFilter:
    if sigma <= 0:
        raise ValueError("--residual-filter-sigma doit etre strictement positif")
    if threshold_mag is not None and threshold_mag <= 0:
        raise ValueError("--residual-filter-threshold-mag doit etre strictement positif")
    if not 0.0 < max_reject_fraction < 1.0:
        raise ValueError("--residual-filter-max-reject-fraction doit etre dans ]0,1[")

    residuals = np.asarray(fit.residuals, dtype=float)
    center = float(np.nanmedian(residuals))
    abs_deviation = np.abs(residuals - center)
    mad = float(np.nanmedian(abs_deviation))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(residuals, ddof=1)) if residuals.size >= 2 else 0.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1e-12

    requested_threshold = float(threshold_mag) if threshold_mag is not None else sigma * scale
    reject_mask = abs_deviation > requested_threshold
    max_reject = int(np.floor(residuals.size * max_reject_fraction))
    if max_reject <= 0:
        reject_mask = np.zeros(residuals.size, dtype=bool)
    elif int(np.sum(reject_mask)) > max_reject:
        reject_mask = np.zeros(residuals.size, dtype=bool)
        rejected_indices = np.argsort(abs_deviation)[-max_reject:]
        reject_mask[rejected_indices] = True

    keep_mask = ~reject_mask
    if int(np.sum(keep_mask)) < min_points:
        raise RuntimeError(
            f"Filtrage robuste des residus abandonne: {int(np.sum(keep_mask))} points conserves, "
            f"minimum requis {min_points}"
        )

    effective_threshold = float(np.min(abs_deviation[reject_mask])) if np.any(reject_mask) else requested_threshold
    return ResidualFilter(
        keep_mask=keep_mask,
        reject_mask=reject_mask,
        residual_center=center,
        residual_scale=float(scale),
        threshold_mag=effective_threshold,
        requested_threshold_mag=requested_threshold,
    )


def write_residual_filter_summary(path: Path, curve: LightCurve, result: ResidualFilter, sigma: float) -> None:
    rejected = int(np.sum(result.reject_mask))
    kept = int(np.sum(result.keep_mask))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "n_points_initial",
                "n_points_kept",
                "n_points_rejected",
                "rejected_fraction",
                "sigma_threshold",
                "residual_center_mag",
                "robust_residual_sigma_mag",
                "requested_threshold_mag",
                "effective_threshold_mag",
            ]
        )
        writer.writerow(
            [
                curve.n_points,
                kept,
                rejected,
                f"{rejected / curve.n_points:.6f}",
                f"{sigma:.6f}",
                f"{result.residual_center:.6f}",
                f"{result.residual_scale:.6f}",
                f"{result.requested_threshold_mag:.6f}",
                f"{result.threshold_mag:.6f}",
            ]
        )


def write_residual_table(
    path: Path,
    curve: LightCurve,
    fit,
    hjd_correction: HjdCorrection | None,
    source_mask: np.ndarray | None = None,
) -> None:
    aligned_mag = aligned_magnitudes(curve, fit)
    aligned_fit = aligned_model(curve, fit)
    if source_mask is None:
        if hjd_correction is None:
            jd_utc_values = curve.jd
            hjd_utc_values = curve.jd
            correction_values = np.zeros_like(curve.jd)
        else:
            jd_utc_values = hjd_correction.jd_utc
            hjd_utc_values = hjd_correction.hjd_utc
            correction_values = hjd_correction.correction_days
    else:
        if hjd_correction is None:
            jd_utc_values = curve.jd
            hjd_utc_values = curve.jd
            correction_values = np.zeros_like(curve.jd)
        else:
            jd_utc_values = hjd_correction.jd_utc[source_mask]
            hjd_utc_values = hjd_correction.hjd_utc[source_mask]
            correction_values = hjd_correction.correction_days[source_mask]

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "jd_utc",
                "hjd_utc",
                "hjd_correction_days",
                "hjd_correction_seconds",
                "magnitude",
                "aligned_magnitude",
                "mag_error",
                "raw_model",
                "aligned_model",
                "residual",
                "file",
            ]
        )
        for jd_utc, hjd_utc, correction_days, mag, aligned_mag_value, err, raw_model, aligned_model_value, residual, group in zip(
            jd_utc_values,
            hjd_utc_values,
            correction_values,
            curve.magnitude,
            aligned_mag,
            curve.mag_error,
            fit.model,
            aligned_fit,
            fit.residuals,
            curve.group,
        ):
            writer.writerow(
                [
                    f"{jd_utc:.8f}",
                    f"{hjd_utc:.8f}",
                    f"{correction_days:.10f}",
                    f"{correction_days * 86400.0:.6f}",
                    f"{mag:.6f}",
                    f"{aligned_mag_value:.6f}",
                    f"{err:.6f}",
                    f"{raw_model:.6f}",
                    f"{aligned_model_value:.6f}",
                    f"{residual:.6f}",
                    curve.group_names[int(group)],
                ]
            )


def run_period_search(
    curve: LightCurve,
    args: argparse.Namespace,
    outdir: Path,
    prefix: str = "",
) -> tuple:
    if args.min_period is None or args.max_period is None:
        raise SystemExit("--min-period et --max-period sont requis sans --period")
    periods = period_grid(args.min_period, args.max_period, args.samples)

    gls = gls_power(curve, periods)
    gls_best_period = float(periods[int(np.argmax(gls))])
    plot_periodogram(
        periods,
        gls,
        gls_best_period,
        "Puissance GLS",
        outdir / f"{prefix}gls_periodogram.png",
    )

    best, candidate_fits, gls_candidates, selection = search_period_order_candidates(
        curve,
        periods,
        gls,
        parse_orders(args.orders),
        top_n=args.gls_candidates,
        multipliers=args.gls_multipliers,
        refine_width=args.refine_width,
        refine_samples=args.candidate_refine_samples,
    )
    best = refine_period(
        curve,
        best.period_days,
        best.order,
        width_fraction=args.refine_width,
        samples=args.refine_samples,
        min_period_days=args.min_period,
        max_period_days=args.max_period,
    )
    selection.selected_fit = best
    order_bests = []
    for order in parse_orders(args.orders):
        fits_for_order = [candidate_fit.fit for candidate_fit in candidate_fits if candidate_fit.fit.order == order]
        if fits_for_order:
            order_bests.append(min(fits_for_order, key=lambda fit: fit.bic))

    bic_score = np.full_like(periods, np.nan, dtype=float)
    for idx, period in enumerate(periods):
        bic_score[idx] = -search_fourier(curve, np.asarray([period]), range(best.order, best.order + 1))[0].bic
    plot_periodogram(
        periods,
        bic_score,
        best.period_days,
        f"Score Fourier ordre {best.order} (-BIC)",
        outdir / f"{prefix}fourier_period_search.png",
    )
    return best, order_bests, candidate_fits, gls_candidates, gls_best_period, gls, selection


def write_ephemeris_by_file(
    path: Path,
    curve,
    ephemerides: list[FileEphemeris],
    hjd_correction: HjdCorrection | None,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "file",
                "label",
                "observer",
                "object_command",
                "mid_jd",
                "ra_deg_icrf",
                "dec_deg_icrf",
                "hjd_correction_mean_s",
                "hjd_correction_min_s",
                "hjd_correction_max_s",
                "source",
            ]
        )
        for group_id, eph in enumerate(ephemerides):
            if hjd_correction is None:
                mean_s = min_s = max_s = ""
            else:
                values = hjd_correction.correction_days[curve.group == group_id] * 86400.0
                mean_s = f"{float(np.mean(values)):.6f}"
                min_s = f"{float(np.min(values)):.6f}"
                max_s = f"{float(np.max(values)):.6f}"
            writer.writerow(
                [
                    eph.file,
                    eph.label,
                    eph.observer,
                    eph.object_command,
                    f"{eph.mid_jd:.8f}",
                    f"{eph.ra_deg:.9f}",
                    f"{eph.dec_deg:.9f}",
                    mean_s,
                    min_s,
                    max_s,
                    "JPL Horizons geocentric",
                ]
            )


def cmd_inspect(args: argparse.Namespace) -> int:
    paths = expand_inputs(args.files)
    curve = read_lightcurve(paths, use_mid_exposure=not args.keep_start_time)
    print(f"Fichiers: {len(curve.files)}")
    print(f"Mesures: {curve.n_points}")
    print(f"Baseline: {curve.baseline_days:.6f} jours")
    for obs in curve.files:
        print(
            f"- {obs.path.name}: {obs.jd.size} mesures, "
            f"objet={obs.object_name or '?'}, filtre={obs.filter_name or '?'}, "
            f"pose={obs.exposure_seconds:g}s"
        )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    paths = expand_inputs(args.files)
    curve = read_lightcurve(paths, use_mid_exposure=not args.keep_start_time)
    outdir = ensure_outdir(args.out)
    fixed_period_days = args.period
    produced_files: list[str] = []
    ephemerides: list[FileEphemeris] = []
    hjd_correction: HjdCorrection | None = None
    if args.residual_filter and fixed_period_days is not None:
        raise SystemExit("--residual-filter necessite une recherche de periode, pas --period")

    if not args.no_ephemeris:
        print("Recuperation des positions RA/DEC par fichier via JPL Horizons...")
        ephemerides = fetch_file_ephemerides(curve, timeout=args.horizons_timeout)
        hjd_correction = apply_hjd_correction(curve, ephemerides)

    selection: PeriodSelection | None = None
    if fixed_period_days is None:
        best, order_bests, candidate_fits, gls_candidates, gls_best_period, gls, selection = run_period_search(
            curve,
            args,
            outdir,
        )
        produced_files.extend(["gls_periodogram.png", "fourier_period_search.png"])
    else:
        if fixed_period_days <= 0:
            raise SystemExit("--period doit etre strictement positif")
        best, order_bests = search_fourier(
            curve,
            np.asarray([fixed_period_days], dtype=float),
            parse_orders(args.orders),
        )
        candidate_fits = []
        gls_candidates = []

    raw_uncertainty = estimate_period_uncertainty(curve, best, delta_chi2=1.0)
    scaled_delta = max(1.0, best.reduced_chi2)
    scaled_uncertainty = estimate_period_uncertainty(curve, best, delta_chi2=scaled_delta)
    amplitude_mag = model_amplitude(best)
    plot_folded_lightcurve(
        curve,
        best,
        outdir / "folded_lightcurve.png",
        period_uncertainty_days=scaled_uncertainty.symmetric_days,
    )
    plot_folded_lightcurve(
        curve,
        best,
        outdir / "folded_lightcurve_by_file.png",
        by_file=True,
        period_uncertainty_days=scaled_uncertainty.symmetric_days,
    )
    plot_folded_lightcurve_by_file_with_residuals(
        curve,
        best,
        outdir / "folded_lightcurve_by_file_with_residuals.png",
        period_uncertainty_days=scaled_uncertainty.symmetric_days,
    )
    plot_residuals(curve, best, outdir / "residuals.png")
    write_period_summary(outdir / "period_summary.csv", best.period_days, amplitude_mag, raw_uncertainty, scaled_uncertainty)
    write_file_summary(outdir / "file_summary.csv", curve, best)
    if fixed_period_days is None:
        write_candidate_summary(outdir / "period_order_candidates.csv", candidate_fits)
        if selection is not None:
            write_period_selection_summary(outdir / "period_selection_summary.csv", selection)
    if ephemerides:
        write_ephemeris_by_file(outdir / "ephemeris_by_file.csv", curve, ephemerides, hjd_correction)
    produced_files.extend(
        [
            "folded_lightcurve.png",
            "folded_lightcurve_by_file.png",
            "folded_lightcurve_by_file_with_residuals.png",
            "residuals.png",
            "residuals.csv",
            "period_summary.csv",
            "file_summary.csv",
            "fourier_order_summary.csv",
        ]
    )
    if fixed_period_days is None:
        produced_files.append("period_order_candidates.csv")
        produced_files.append("period_selection_summary.csv")
    if ephemerides:
        produced_files.append("ephemeris_by_file.csv")
    write_residual_table(outdir / "residuals.csv", curve, best, hjd_correction)

    write_order_summary(outdir / "fourier_order_summary.csv", order_bests)

    residual_filter_result: ResidualFilter | None = None
    residual_best = None
    residual_amplitude_mag: float | None = None
    if args.residual_filter:
        residual_filter_result = residual_filter_from_fit(
            best,
            sigma=args.residual_filter_sigma,
            threshold_mag=args.residual_filter_threshold_mag,
            max_reject_fraction=args.residual_filter_max_reject_fraction,
            min_points=args.residual_filter_min_points,
        )
        write_residual_filter_summary(
            outdir / "residual_filter_summary.csv",
            curve,
            residual_filter_result,
            args.residual_filter_sigma,
        )
        plot_residual_filter(
            curve,
            best,
            residual_filter_result.reject_mask,
            residual_filter_result.residual_center,
            residual_filter_result.threshold_mag,
            outdir / "residual_filter_rejected_points.png",
        )
        produced_files.append("residual_filter_summary.csv")
        produced_files.append("residual_filter_rejected_points.png")

        if np.any(residual_filter_result.reject_mask):
            filtered_curve = subset_lightcurve(curve, residual_filter_result.keep_mask)
            (
                residual_best,
                residual_order_bests,
                residual_candidate_fits,
                _residual_gls_candidates,
                _residual_gls_best_period,
                _residual_gls,
                residual_selection,
            ) = run_period_search(filtered_curve, args, outdir, prefix="residual_filtered_")
            residual_raw_uncertainty = estimate_period_uncertainty(filtered_curve, residual_best, delta_chi2=1.0)
            residual_scaled_delta = max(1.0, residual_best.reduced_chi2)
            residual_scaled_uncertainty = estimate_period_uncertainty(
                filtered_curve,
                residual_best,
                delta_chi2=residual_scaled_delta,
            )
            residual_amplitude_mag = model_amplitude(residual_best)
            plot_folded_lightcurve(
                filtered_curve,
                residual_best,
                outdir / "residual_filtered_folded_lightcurve.png",
                period_uncertainty_days=residual_scaled_uncertainty.symmetric_days,
            )
            plot_folded_lightcurve(
                filtered_curve,
                residual_best,
                outdir / "residual_filtered_folded_lightcurve_by_file.png",
                by_file=True,
                period_uncertainty_days=residual_scaled_uncertainty.symmetric_days,
            )
            plot_folded_lightcurve_by_file_with_residuals(
                filtered_curve,
                residual_best,
                outdir / "residual_filtered_folded_lightcurve_by_file_with_residuals.png",
                period_uncertainty_days=residual_scaled_uncertainty.symmetric_days,
            )
            plot_residuals(filtered_curve, residual_best, outdir / "residual_filtered_residuals.png")
            write_period_summary(
                outdir / "residual_filtered_period_summary.csv",
                residual_best.period_days,
                residual_amplitude_mag,
                residual_raw_uncertainty,
                residual_scaled_uncertainty,
            )
            write_file_summary(outdir / "residual_filtered_file_summary.csv", filtered_curve, residual_best)
            write_candidate_summary(outdir / "residual_filtered_period_order_candidates.csv", residual_candidate_fits)
            write_order_summary(outdir / "residual_filtered_fourier_order_summary.csv", residual_order_bests)
            write_period_selection_summary(outdir / "residual_filtered_period_selection_summary.csv", residual_selection)
            write_residual_table(
                outdir / "residual_filtered_residuals.csv",
                filtered_curve,
                residual_best,
                hjd_correction,
                source_mask=residual_filter_result.keep_mask,
            )
            produced_files.extend(
                [
                    "residual_filtered_gls_periodogram.png",
                    "residual_filtered_fourier_period_search.png",
                    "residual_filtered_folded_lightcurve.png",
                    "residual_filtered_folded_lightcurve_by_file.png",
                    "residual_filtered_folded_lightcurve_by_file_with_residuals.png",
                    "residual_filtered_residuals.png",
                    "residual_filtered_residuals.csv",
                    "residual_filtered_period_summary.csv",
                    "residual_filtered_file_summary.csv",
                    "residual_filtered_period_order_candidates.csv",
                    "residual_filtered_fourier_order_summary.csv",
                    "residual_filtered_period_selection_summary.csv",
                ]
            )

    produced_files.append("run_metadata.json")
    write_run_metadata(outdir / "run_metadata.json", args, paths, curve, produced_files, best, residual_best, selection)

    print("=== Donnees ===")
    print(f"Fichiers: {len(curve.files)}")
    print(f"Mesures: {curve.n_points}")
    print(f"Baseline: {curve.baseline_days:.6f} jours")
    print()
    if fixed_period_days is None:
        print("=== GLS ===")
        print(f"Meilleure periode GLS: {format_period(gls_best_period)}")
        if gls_best_period * 2.0 <= args.max_period:
            print(f"Candidat double-pic 2 x GLS: {format_period(gls_best_period * 2.0)}")
        print(f"Puissance GLS: {float(np.max(gls)):.6f}")
        print("Note: pour un asteroide, GLS peut accrocher P/2 si la courbe est double-pic.")
        print()
    else:
        print("=== Periode imposee ===")
        print(f"Periode: {format_period(fixed_period_days)}")
        print("Recherche GLS/Fourier sautee; ajustement Fourier seulement a la periode imposee.")
        print()
    print("=== Fourier ===")
    print(f"Meilleure periode: {format_period(best.period_days)}")
    print(f"Ordre retenu: {best.order}")
    print(f"Chi2 reduit: {best.reduced_chi2:.6f}")
    print(f"Amplitude: {amplitude_mag:.3f} mag")
    print(f"AIC: {best.aic:.6f}")
    print(f"AICc: {best.aicc:.6f}")
    print(f"BIC: {best.bic:.6f}")
    if selection is not None:
        print(f"Strategie de selection: {selection.strategy}")
        print(
            "Reference ordre bas: "
            f"ordre {selection.reference_fit.order}, {format_period(selection.reference_fit.period_days)}"
        )
        if (
            abs(selection.bic_best_fit.period_days - best.period_days)
            > selection.stability_tolerance * best.period_days
        ):
            print(
                "Meilleur BIC brut non retenu: "
                f"{format_period(selection.bic_best_fit.period_days)}, "
                f"ordre {selection.bic_best_fit.order}, BIC {selection.bic_best_fit.bic:.6f}"
            )
    if fixed_period_days is None:
        print(f"Periodes candidates GLS: {len(gls_candidates)}")
        print(f"Couples periode/ordre testes: {len(candidate_fits)}")
    print()
    print("=== Incertitude sur P ===")
    print(
        "Profil chi2, delta chi2 = 1: "
        f"-{format_uncertainty(raw_uncertainty.minus_days)} / "
        f"+{format_uncertainty(raw_uncertainty.plus_days)}"
    )
    print(
        f"Profil chi2 reechelonne, delta chi2 = {scaled_delta:.6f}: "
        f"-{format_uncertainty(scaled_uncertainty.minus_days)} / "
        f"+{format_uncertainty(scaled_uncertainty.plus_days)}"
    )
    if scaled_delta > 1.0:
        print("Note: l'incertitude reechelonnee tient compte du chi2 reduit > 1.")
    print()
    if residual_filter_result is not None:
        print("=== Filtrage robuste des residus ===")
        print(
            f"Points rejetes: {int(np.sum(residual_filter_result.reject_mask))} / "
            f"{residual_filter_result.reject_mask.size}"
        )
        print(
            "Seuil residu: "
            f"{residual_filter_result.threshold_mag:.6f} mag "
            f"({args.residual_filter_sigma:.2f} sigma robustes)"
        )
        if residual_best is None:
            print("Aucune nouvelle recherche lancee: aucun point rejete.")
        else:
            print(f"Periode filtree: {format_period(residual_best.period_days)}")
            print(f"Ordre filtre: {residual_best.order}")
            if residual_amplitude_mag is not None:
                print(f"Amplitude filtree: {residual_amplitude_mag:.3f} mag")
            print(f"BIC filtre: {residual_best.bic:.6f}")
        print()
    print("=== Fichiers produits ===")
    for name in produced_files:
        print(outdir / name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asteroid-lc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Resumer les fichiers de mesures")
    inspect_parser.add_argument("files", nargs="+", help="Fichiers ou motifs glob")
    inspect_parser.add_argument("--keep-start-time", action="store_true", help="Ne pas convertir vers le milieu de pose")
    inspect_parser.set_defaults(func=cmd_inspect)

    search_parser = subparsers.add_parser("search", help="Chercher la periode de rotation")
    search_parser.add_argument("files", nargs="+", help="Fichiers ou motifs glob")
    search_parser.add_argument("--min-period", type=float, help="Periode minimale en jours")
    search_parser.add_argument("--max-period", type=float, help="Periode maximale en jours")
    search_parser.add_argument(
        "--period",
        type=float,
        help="Periode imposee en jours; saute la recherche de periode",
    )
    search_parser.add_argument("--samples", type=int, default=8000, help="Nombre d'echantillons de periode")
    search_parser.add_argument("--orders", default="1:12", help="Ordres Fourier, ex: 1:12 ou 4")
    search_parser.add_argument("--gls-candidates", type=int, default=20, help="Nombre de pics GLS a tester")
    search_parser.add_argument(
        "--gls-multipliers",
        type=parse_multipliers,
        default=(0.5, 1.0, 2.0),
        help="Multiplicateurs appliques a chaque pic GLS, ex: 0.5,1,2",
    )
    search_parser.add_argument("--refine-width", type=float, default=0.01, help="Demi-largeur relative du raffinement")
    search_parser.add_argument(
        "--candidate-refine-samples",
        type=int,
        default=300,
        help="Echantillons de raffinement par couple candidat periode/ordre",
    )
    search_parser.add_argument("--refine-samples", type=int, default=2000, help="Echantillons du raffinement")
    search_parser.add_argument(
        "--residual-filter",
        action="store_true",
        help="Apres la recherche initiale, rejeter les forts residus et relancer une recherche de periode",
    )
    search_parser.add_argument(
        "--residual-filter-sigma",
        type=float,
        default=3.5,
        help="Seuil robuste en sigma MAD pour rejeter les residus",
    )
    search_parser.add_argument(
        "--residual-filter-threshold-mag",
        type=float,
        help="Seuil absolu de residu en magnitude; remplace le seuil sigma",
    )
    search_parser.add_argument(
        "--residual-filter-max-reject-fraction",
        type=float,
        default=0.25,
        help="Fraction maximale de points rejetes par le filtrage robuste des residus",
    )
    search_parser.add_argument(
        "--residual-filter-min-points",
        type=int,
        default=30,
        help="Nombre minimal de points a conserver apres filtrage robuste des residus",
    )
    search_parser.add_argument("--keep-start-time", action="store_true", help="Ne pas convertir vers le milieu de pose")
    search_parser.add_argument(
        "--no-ephemeris",
        action="store_true",
        help="Ne pas interroger JPL Horizons pour les positions RA/DEC par fichier",
    )
    search_parser.add_argument(
        "--horizons-timeout",
        type=float,
        default=30.0,
        help="Timeout en secondes pour les appels JPL Horizons",
    )
    search_parser.add_argument("--out", default="output", help="Dossier de sortie")
    search_parser.set_defaults(func=cmd_search)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
