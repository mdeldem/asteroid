from pathlib import Path

import numpy as np

from asteroid_lightcurve.ephemeris import FileCorrection, FileEphemeris, cache_paths, write_file_correction_cache
from asteroid_lightcurve.models import LightCurve, ObservationFile
from asteroid_lightcurve.period import FitResult, design_matrix, weighted_fit
from asteroid_lightcurve.plotting import aligned_magnitudes, aligned_model, plot_magnitudes


def make_curve(jd, mag, group, mag_observed=None) -> LightCurve:
    files = [
        ObservationFile(path=Path(f"file_{idx}.txt"), fmt="DVv")
        for idx in range(int(np.max(group)) + 1)
    ]
    return LightCurve(
        jd=np.asarray(jd, dtype=float),
        jd_decimals=np.full(len(jd), 8, dtype=int),
        magnitude=np.asarray(mag, dtype=float),
        mag_error=np.full(len(jd), 0.02, dtype=float),
        group=np.asarray(group, dtype=int),
        group_names=[file.path.name for file in files],
        files=files,
        mag_observed=np.asarray(mag_observed if mag_observed is not None else mag, dtype=float),
    )


def test_design_matrix_has_pravec_file_offsets_before_fourier() -> None:
    jd = np.array([2459000.0, 2459000.1, 2459000.2, 2459000.3])
    groups = np.array([0, 1, 2, 1])

    matrix = design_matrix(jd, groups, period_days=0.5, order=2, n_groups=3)

    assert matrix.shape == (4, 7)
    np.testing.assert_allclose(matrix[:, 0], 1.0)
    np.testing.assert_array_equal(matrix[:, 1], groups == 1)
    np.testing.assert_array_equal(matrix[:, 2], groups == 2)


def test_offsets_are_fit_simultaneously_not_precomputed_by_file_median() -> None:
    period_days = 1.0
    phase0 = np.linspace(0.00, 0.18, 20)
    phase1 = np.linspace(0.50, 0.68, 20)
    jd = np.concatenate([phase0, phase1])
    groups = np.concatenate([np.zeros(phase0.size, dtype=int), np.ones(phase1.size, dtype=int)])
    rotation = 15.0 + 0.4 * np.cos(2.0 * np.pi * jd)
    true_offset = 0.35
    mag = rotation + true_offset * (groups == 1)
    curve = make_curve(jd, mag, groups)

    median_offset = float(np.median(mag[groups == 1]) - np.median(mag[groups == 0]))
    fit = weighted_fit(curve, period_days=period_days, order=1)

    assert abs(median_offset - true_offset) > 0.4
    assert abs(float(fit.coefficients[1]) - true_offset) < 1e-10


def test_correction_cache_contains_no_fit_dependent_offsets(tmp_path: Path) -> None:
    source = tmp_path / "obs_001.txt"
    source.write_text("FMT DVv\n2459000.0 12.0 0.02\n", encoding="utf-8")
    values = np.array([2459000.0, 2459000.1])
    correction = FileCorrection(
        jd_utc=values,
        hjd_utc=values + 0.001,
        light_time_correction_days=np.full(2, 0.001),
        mag_obs=np.array([12.0, 12.1]),
        mag_error=np.full(2, 0.02),
        r_au=np.full(2, 2.1),
        delta_au=np.full(2, 1.1),
        geometry_correction=np.full(2, 5.0 * np.log10(2.1 * 1.1)),
        mag_reduced=np.array([10.18, 10.28]),
        ephemeris=FileEphemeris(
            file=source.name,
            label="[2022-01-01]",
            observer="obs",
            object_command="250;",
            mid_jd=float(np.mean(values)),
            ra_deg=1.0,
            dec_deg=2.0,
            r_au=2.1,
            delta_au=1.1,
        ),
    )

    write_file_correction_cache(source, "250;", correction)
    csv_path, json_path = cache_paths(source)
    cache_text = csv_path.read_text(encoding="utf-8")
    meta_text = json_path.read_text(encoding="utf-8")

    assert "mag_reduced" in cache_text
    for forbidden in ["block_offset", "Zi", "mag_plot", "residuals"]:
        assert forbidden not in cache_text
        assert forbidden not in meta_text


def test_plot_shift_is_global_and_preserves_residuals() -> None:
    curve = make_curve(
        jd=[0.0, 0.25, 0.5, 0.75],
        mag=[7.8, 7.9, 7.7, 7.85],
        group=[0, 0, 1, 1],
        mag_observed=[11.7, 11.8, 11.6, 11.75],
    )
    fit = FitResult(
        period_days=1.0,
        order=1,
        chi2=1.0,
        reduced_chi2=1.0,
        bic=1.0,
        aic=1.0,
        aicc=1.0,
        harmonic_significance=(1.0,),
        coefficients=np.array([7.8, -0.1, 0.05, 0.0]),
        model=np.array([7.85, 7.8, 7.75, 7.8]),
        residuals=np.array([-0.05, 0.1, -0.05, 0.05]),
    )

    mag_plot, model_plot, shift = plot_magnitudes(curve, fit)
    aligned_mag = aligned_magnitudes(curve, fit)
    aligned_fit = aligned_model(curve, fit)

    assert np.isscalar(shift)
    assert abs(float(np.median(mag_plot)) - float(np.median(curve.mag_observed))) < 1e-12
    np.testing.assert_allclose(mag_plot - model_plot, aligned_mag - aligned_fit)


def test_plot_shift_is_not_per_file_recentering() -> None:
    curve = make_curve(
        jd=[0.0, 0.1, 0.5, 0.6],
        mag=[8.0, 8.1, 7.0, 7.1],
        group=[0, 0, 1, 1],
        mag_observed=[12.0, 12.1, 12.2, 12.3],
    )
    fit = FitResult(
        period_days=1.0,
        order=0,
        chi2=1.0,
        reduced_chi2=1.0,
        bic=1.0,
        aic=1.0,
        aicc=1.0,
        harmonic_significance=(),
        coefficients=np.array([8.0, -1.0]),
        model=np.array([8.0, 8.0, 7.0, 7.0]),
        residuals=np.array([0.0, 0.1, 0.0, 0.1]),
    )

    mag_plot, _model_plot, shift = plot_magnitudes(curve, fit)

    group_shifts = mag_plot - aligned_magnitudes(curve, fit)
    np.testing.assert_allclose(group_shifts, np.full(curve.n_points, shift))
