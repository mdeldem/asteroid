from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .parser import expand_inputs, read_lightcurve
from .period import PeriodUncertainty, estimate_period_uncertainty, gls_power, period_grid, refine_period, search_fourier
from .plotting import (
    corrected_magnitudes,
    corrected_model,
    ensure_outdir,
    format_period,
    plot_folded_lightcurve,
    plot_periodogram,
    plot_residuals,
)


def parse_orders(value: str) -> range:
    if ":" in value:
        start, end = value.split(":", 1)
        return range(int(start), int(end) + 1)
    order = int(value)
    return range(order, order + 1)


def format_uncertainty(value_days: float | None) -> str:
    if value_days is None:
        return "non bornee"
    return f"{value_days * 24.0:.6f} h ({value_days:.8f} j)"


def write_period_summary(
    path: Path,
    best_period_days: float,
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

    if fixed_period_days is None:
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
            outdir / "gls_periodogram.png",
        )
        produced_files.append("gls_periodogram.png")

        best, order_bests = search_fourier(curve, periods, parse_orders(args.orders))
        best = refine_period(
            curve,
            best.period_days,
            best.order,
            width_fraction=args.refine_width,
            samples=args.refine_samples,
        )

        bic_score = np.full_like(periods, np.nan, dtype=float)
        for idx, period in enumerate(periods):
            bic_score[idx] = -search_fourier(curve, np.asarray([period]), range(best.order, best.order + 1))[0].bic
        plot_periodogram(
            periods,
            bic_score,
            best.period_days,
            f"Score Fourier ordre {best.order} (-BIC)",
            outdir / "fourier_period_search.png",
        )
        produced_files.append("fourier_period_search.png")
    else:
        if fixed_period_days <= 0:
            raise SystemExit("--period doit etre strictement positif")
        best, order_bests = search_fourier(
            curve,
            np.asarray([fixed_period_days], dtype=float),
            parse_orders(args.orders),
        )

    raw_uncertainty = estimate_period_uncertainty(curve, best, delta_chi2=1.0)
    scaled_delta = max(1.0, best.reduced_chi2)
    scaled_uncertainty = estimate_period_uncertainty(curve, best, delta_chi2=scaled_delta)
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
    plot_residuals(curve, best, outdir / "residuals.png")
    write_period_summary(outdir / "period_summary.csv", best.period_days, raw_uncertainty, scaled_uncertainty)
    produced_files.extend(
        [
            "folded_lightcurve.png",
            "folded_lightcurve_by_file.png",
            "residuals.png",
            "residuals.csv",
            "period_summary.csv",
            "fourier_order_summary.csv",
        ]
    )
    residual_path = outdir / "residuals.csv"
    corrected_mag = corrected_magnitudes(curve, best)
    corrected_fit = corrected_model(curve, best)
    with residual_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "jd",
                "magnitude",
                "corrected_magnitude",
                "mag_error",
                "model",
                "corrected_model",
                "residual",
                "file",
            ]
        )
        for jd, mag, corr_mag, err, model, corr_model, residual, group in zip(
            curve.jd,
            curve.magnitude,
            corrected_mag,
            curve.mag_error,
            best.model,
            corrected_fit,
            best.residuals,
            curve.group,
        ):
            writer.writerow(
                [
                    f"{jd:.8f}",
                    f"{mag:.6f}",
                    f"{corr_mag:.6f}",
                    f"{err:.6f}",
                    f"{model:.6f}",
                    f"{corr_model:.6f}",
                    f"{residual:.6f}",
                    curve.group_names[int(group)],
                ]
            )

    with (outdir / "fourier_order_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["order", "period_days", "period_hours", "chi2", "reduced_chi2", "aic", "bic"])
        for fit in order_bests:
            writer.writerow(
                [
                    fit.order,
                    f"{fit.period_days:.10f}",
                    f"{fit.period_hours:.10f}",
                    f"{fit.chi2:.6f}",
                    f"{fit.reduced_chi2:.6f}",
                    f"{fit.aic:.6f}",
                    f"{fit.bic:.6f}",
                ]
            )

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
    print(f"AIC: {best.aic:.6f}")
    print(f"BIC: {best.bic:.6f}")
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
    search_parser.add_argument("--refine-width", type=float, default=0.01, help="Demi-largeur relative du raffinement")
    search_parser.add_argument("--refine-samples", type=int, default=2000, help="Echantillons du raffinement")
    search_parser.add_argument("--keep-start-time", action="store_true", help="Ne pas convertir vers le milieu de pose")
    search_parser.add_argument("--out", default="output", help="Dossier de sortie")
    search_parser.set_defaults(func=cmd_search)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
