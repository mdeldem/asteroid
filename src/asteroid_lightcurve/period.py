from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import brentq

from .models import LightCurve


@dataclass
class FitResult:
    period_days: float
    order: int
    chi2: float
    reduced_chi2: float
    bic: float
    aic: float
    coefficients: np.ndarray
    model: np.ndarray
    residuals: np.ndarray

    @property
    def period_hours(self) -> float:
        return self.period_days * 24.0


@dataclass
class PeriodUncertainty:
    period_days: float
    delta_chi2: float
    lower_days: float | None
    upper_days: float | None

    @property
    def period_hours(self) -> float:
        return self.period_days * 24.0

    @property
    def lower_hours(self) -> float | None:
        if self.lower_days is None:
            return None
        return self.lower_days * 24.0

    @property
    def upper_hours(self) -> float | None:
        if self.upper_days is None:
            return None
        return self.upper_days * 24.0

    @property
    def minus_days(self) -> float | None:
        if self.lower_days is None:
            return None
        return self.period_days - self.lower_days

    @property
    def plus_days(self) -> float | None:
        if self.upper_days is None:
            return None
        return self.upper_days - self.period_days

    @property
    def symmetric_days(self) -> float | None:
        values = [value for value in [self.minus_days, self.plus_days] if value is not None]
        if not values:
            return None
        return float(np.mean(values))

    @property
    def minus_hours(self) -> float | None:
        if self.minus_days is None:
            return None
        return self.minus_days * 24.0

    @property
    def plus_hours(self) -> float | None:
        if self.plus_days is None:
            return None
        return self.plus_days * 24.0

    @property
    def symmetric_hours(self) -> float | None:
        if self.symmetric_days is None:
            return None
        return self.symmetric_days * 24.0


def period_grid(min_period_days: float, max_period_days: float, samples: int) -> np.ndarray:
    if min_period_days <= 0 or max_period_days <= min_period_days:
        raise ValueError("Les bornes de periode doivent verifier 0 < min < max")
    frequency = np.linspace(1.0 / max_period_days, 1.0 / min_period_days, samples)
    return 1.0 / frequency


def _weights(errors: np.ndarray) -> np.ndarray:
    safe_errors = np.asarray(errors, dtype=float)
    finite = np.isfinite(safe_errors) & (safe_errors > 0)
    if not np.all(finite):
        replacement = np.nanmedian(safe_errors[finite]) if np.any(finite) else 1.0
        safe_errors = np.where(finite, safe_errors, replacement)
    return 1.0 / np.square(safe_errors)


def design_matrix(
    jd: np.ndarray,
    groups: np.ndarray,
    period_days: float,
    order: int,
    n_groups: int,
) -> np.ndarray:
    t = np.asarray(jd, dtype=float) - float(np.min(jd))
    phase_arg = 2.0 * np.pi * t / period_days
    columns = [np.ones_like(t)]
    for group_id in range(1, n_groups):
        columns.append((groups == group_id).astype(float))
    for harmonic in range(1, order + 1):
        columns.append(np.cos(harmonic * phase_arg))
        columns.append(np.sin(harmonic * phase_arg))
    return np.column_stack(columns)


def weighted_fit(
    curve: LightCurve,
    period_days: float,
    order: int,
    per_group_offsets: bool = True,
) -> FitResult:
    n_groups = len(curve.group_names) if per_group_offsets else 1
    groups = curve.group if per_group_offsets else np.zeros_like(curve.group)
    x = design_matrix(curve.jd, groups, period_days, order, n_groups)
    y = curve.magnitude
    w_sqrt = np.sqrt(_weights(curve.mag_error))
    xw = x * w_sqrt[:, None]
    yw = y * w_sqrt
    coefficients, *_ = lstsq(xw, yw)
    model = x @ coefficients
    residuals = y - model
    chi2 = float(np.sum(np.square(residuals * w_sqrt)))
    n = y.size
    k = x.shape[1]
    dof = max(n - k, 1)
    reduced = chi2 / dof
    aic = chi2 + 2.0 * k
    bic = chi2 + k * np.log(max(n, 2))
    return FitResult(
        period_days=float(period_days),
        order=order,
        chi2=chi2,
        reduced_chi2=float(reduced),
        bic=float(bic),
        aic=float(aic),
        coefficients=coefficients,
        model=model,
        residuals=residuals,
    )


def gls_power(curve: LightCurve, periods_days: np.ndarray) -> np.ndarray:
    baseline = weighted_fit(curve, float(periods_days[0]), 0)
    constant_chi2 = baseline.chi2
    powers = np.empty_like(periods_days, dtype=float)
    for idx, period in enumerate(periods_days):
        fit = weighted_fit(curve, float(period), 1)
        powers[idx] = max(0.0, 1.0 - fit.chi2 / constant_chi2) if constant_chi2 > 0 else 0.0
    return powers


def search_fourier(
    curve: LightCurve,
    periods_days: np.ndarray,
    orders: range,
) -> tuple[FitResult, list[FitResult]]:
    best: FitResult | None = None
    all_best: list[FitResult] = []
    for order in orders:
        order_best: FitResult | None = None
        for period in periods_days:
            fit = weighted_fit(curve, float(period), order)
            if order_best is None or fit.bic < order_best.bic:
                order_best = fit
        if order_best is None:
            continue
        all_best.append(order_best)
        if best is None or order_best.bic < best.bic:
            best = order_best
    if best is None:
        raise RuntimeError("Aucun ajustement Fourier n'a pu etre calcule")
    return best, all_best


def refine_period(
    curve: LightCurve,
    initial_period_days: float,
    order: int,
    width_fraction: float = 0.01,
    samples: int = 2000,
) -> FitResult:
    half_width = initial_period_days * width_fraction
    low = max(initial_period_days - half_width, initial_period_days * 0.5)
    high = initial_period_days + half_width
    periods = period_grid(low, high, samples)
    best: FitResult | None = None
    for period in periods:
        fit = weighted_fit(curve, float(period), order)
        if best is None or fit.bic < best.bic:
            best = fit
    if best is None:
        raise RuntimeError("Raffinement impossible")
    return best


def estimate_period_uncertainty(
    curve: LightCurve,
    fit: FitResult,
    delta_chi2: float = 1.0,
    max_fraction: float = 0.25,
) -> PeriodUncertainty:
    if delta_chi2 <= 0:
        raise ValueError("delta_chi2 doit etre strictement positif")

    period = fit.period_days
    target = fit.chi2 + delta_chi2

    def chi2_minus_target(period_days: float) -> float:
        return weighted_fit(curve, period_days, fit.order).chi2 - target

    def find_crossing(direction: int) -> float | None:
        step = max(period * 1e-6, 1e-8)
        previous = period
        limit = period * (1.0 + direction * max_fraction)
        for _ in range(80):
            current = period + direction * step
            if (direction < 0 and current <= limit) or (direction > 0 and current >= limit):
                return None
            value = chi2_minus_target(current)
            if value >= 0:
                low, high = sorted([previous, current])
                return float(brentq(chi2_minus_target, low, high, maxiter=100))
            previous = current
            step *= 1.5
        return None

    return PeriodUncertainty(
        period_days=period,
        delta_chi2=delta_chi2,
        lower_days=find_crossing(-1),
        upper_days=find_crossing(1),
    )
