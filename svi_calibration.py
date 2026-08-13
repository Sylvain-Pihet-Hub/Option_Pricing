import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, minimize


class SVICalibrator:
    """Calibrates a raw-SVI slice (single maturity) to market implied vols
    using non-linear least squares.

    Raw SVI parameterization of total implied variance:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)**2 + sigma**2))
    where k = log(strike / forward) is the log-moneyness and
    w(k) = implied_vol(k)**2 * T is the total variance.
    """

    def __init__(self, strikes: np.ndarray, implied_vols: np.ndarray, forward: float, maturity: float):
        self.strikes = np.asarray(strikes, dtype=float)
        self.implied_vols = np.asarray(implied_vols, dtype=float)
        self.forward = forward
        self.maturity = maturity
        self.log_moneyness = np.log(self.strikes / self.forward)
        self.market_total_variance = self.implied_vols ** 2 * self.maturity
        self.params = None

    @staticmethod
    def total_variance(params: np.ndarray, k: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def _residuals(self, params: np.ndarray) -> np.ndarray:
        return self.total_variance(params, self.log_moneyness) - self.market_total_variance

    def _initial_guess(self) -> np.ndarray:
        w = self.market_total_variance
        k = self.log_moneyness
        a0 = max(w.min() * 0.5, 1e-6)
        b0 = 0.1
        rho0 = 0.0
        m0 = k[np.argmin(w)]
        sigma0 = 0.1
        return np.array([a0, b0, rho0, m0, sigma0])

    def _bounds(self) -> tuple:
        k = self.log_moneyness
        w = self.market_total_variance
        lower = [0.0, 1e-6, -0.999, k.min() - 1.0, 1e-6]
        upper = [w.max() * 2 + 1e-6, 5.0, 0.999, k.max() + 1.0, 5.0]
        return lower, upper

    def calibrate(self, initial_guess: np.ndarray = None) -> np.ndarray:
        x0 = self._initial_guess() if initial_guess is None else np.asarray(initial_guess, dtype=float)
        result = least_squares(self._residuals, x0, bounds=self._bounds())
        self.params = result.x
        return self.params

    def fitted_implied_vol(self, k: np.ndarray = None) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Call calibrate() before requesting fitted implied vols.")
        k = self.log_moneyness if k is None else np.asarray(k, dtype=float)
        w = self.total_variance(self.params, k)
        return np.sqrt(w / self.maturity)


class SSVICalibrator:
    """Calibrates a single global SSVI (Surface SVI) implied volatility surface
    jointly across several maturities, using non-linear least squares.

    Total variance surface (Gatheral & Jacquier):
        w(k, theta) = theta/2 * (1 + rho*phi(theta)*k + sqrt((phi(theta)*k + rho)**2 + (1 - rho**2)))
    with the power-law shape function:
        phi(theta) = eta / (theta**gamma * (1 + theta)**(1 - gamma))

    theta_t is the ATM total variance of each maturity slice (read directly off the
    market smile at k=0), while rho, eta and gamma are the three global parameters
    shared across all maturities and calibrated simultaneously.

    No-arbitrage
    ------------
    Calendar spread: w(k, .) must be non-decreasing in maturity at every k. For the
    power-law shape function, theta*phi(theta) = eta*(theta/(1+theta))**(1-gamma) is
    increasing in theta and d/dtheta[theta*phi] / phi = (1-gamma)/(1+theta) < 1, which is
    below (1+sqrt(1-rho**2))/rho**2 >= 2 for any rho. Theorem 4.1 of Gatheral-Jacquier is
    therefore satisfied for free as long as gamma < 1, and the condition collapses to
    "theta_t non-decreasing in t". That is imposed on the calibrated theta values, and
    preserved between expiries by a shape-preserving PCHIP interpolator, which unlike a
    cubic spline cannot overshoot into a decreasing segment.

    Butterfly (non-negative risk-neutral density), Theorem 4.2 - it is sufficient that
        theta*phi(theta)*(1 + |rho|) < 4    and    theta*phi(theta)**2*(1 + |rho|) <= 4
    for every theta on the surface. These are carried as hard non-linear constraints in
    the calibration, and the fitted slices are checked afterwards through Gatheral's g(k).
    """

    def __init__(self, maturities, strikes_by_maturity, implied_vols_by_maturity, forwards, weights_by_maturity=None, refit_theta: bool = True,
                 enforce_butterfly: bool = True, max_iterations: int = 8, tol: float = 1e-8):
        self.maturities = np.asarray(maturities, dtype=float)
        n_slices = len(self.maturities)
        if not (len(strikes_by_maturity) == n_slices and len(implied_vols_by_maturity) == n_slices and len(forwards) == n_slices):
            raise ValueError("maturities, strikes_by_maturity, implied_vols_by_maturity and forwards must have the same length.")
        if n_slices < 2:
            raise ValueError("SSVI ties several maturities together and needs at least 2 slices.")
        self.forwards = np.asarray(forwards, dtype=float)
        self.refit_theta = refit_theta
        self.enforce_butterfly = enforce_butterfly
        self.max_iterations = max_iterations
        self.tol = tol

        # The theta term structure only makes sense in maturity order, and the calendar
        # condition is stated slice to slice, so the slices are sorted up front.
        slice_order = np.argsort(self.maturities)
        self.maturities = self.maturities[slice_order]
        self.forwards = self.forwards[slice_order]

        self.strikes_by_maturity = []
        self.log_moneyness_by_maturity = []
        self.market_implied_vol_by_maturity = []
        self.market_total_variance_by_maturity = []
        self.weights_by_maturity = []
        self.theta = np.zeros(n_slices)

        for i, slice_index in enumerate(slice_order):
            strikes = np.asarray(strikes_by_maturity[slice_index], dtype=float)
            implied_vols = np.asarray(implied_vols_by_maturity[slice_index], dtype=float)
            k = np.log(strikes / self.forwards[i])
            w = implied_vols ** 2 * self.maturities[i]
            order = np.argsort(k)
            if not (k[order][0] < 0.0 < k[order][-1]):
                warnings.warn(f"Slice T={self.maturities[i]:.4f} has no strikes straddling the "
                              f"forward, so its ATM total variance is extrapolated, not observed.")
            weights = np.ones_like(k) if weights_by_maturity is None else np.asarray(weights_by_maturity[slice_index], dtype=float)
            self.strikes_by_maturity.append(strikes[order])
            self.log_moneyness_by_maturity.append(k[order])
            self.market_implied_vol_by_maturity.append(implied_vols[order])
            self.market_total_variance_by_maturity.append(w[order])
            self.weights_by_maturity.append(weights[order])
            self.theta[i] = np.interp(0.0, k[order], w[order])

        # Residual scale should not depend on the level of spot or on how the weights were
        # produced, so they are normalised to average one across the whole surface.
        mean_weight = np.mean(np.concatenate(self.weights_by_maturity))
        self.weights_by_maturity = [weights / mean_weight for weights in self.weights_by_maturity]

        # A running maximum is the cheapest projection of the observed ATM variances onto a
        # non-decreasing term structure: quoted mid prices are noisy enough that a short
        # expiry can print above the next one purely on quote noise.
        self.theta = np.maximum.accumulate(self.theta)
        self.params = None
        self._build_theta_interpolator()

    @staticmethod
    def phi(theta: np.ndarray, eta: float, gamma: float):
        return eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))

    @classmethod
    def total_variance(cls, params: np.ndarray, theta, k: np.ndarray):
        rho, eta, gamma = params
        phi = cls.phi(theta, eta, gamma)
        return theta / 2.0 * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1.0 - rho ** 2)))

    def _residuals(self, params: np.ndarray):
        residuals = [
            weights * (self.total_variance(params, theta, k) - w_mkt)
            for k, w_mkt, theta, weights in zip(self.log_moneyness_by_maturity, self.market_total_variance_by_maturity, self.theta, self.weights_by_maturity)
        ]
        return np.concatenate(residuals)

    def _initial_guess(self):
        return np.array([-0.5, 1.0, 0.5])

    def _bounds(self):
        lower = [-0.999, 1e-6, 1e-6]
        upper = [0.999, 10.0, 1.0 - 1e-6]
        return lower, upper

    def _theta_grid(self, n: int = 64):
        """Theta values the calibrated surface can take, i.e. where butterfly must hold."""
        return np.geomspace(max(self.theta.min(), 1e-8), max(self.theta.max(), 1e-7), n)

    def butterfly_margins(self, params: np.ndarray, theta_grid: np.ndarray = None):
        """Slack in the two butterfly conditions of Theorem 4.2; negative means arbitrage."""
        rho, eta, gamma = params
        theta_grid = self._theta_grid() if theta_grid is None else theta_grid
        phi = self.phi(theta_grid, eta, gamma)
        return np.concatenate([
            4.0 - theta_grid * phi * (1.0 + abs(rho)),
            4.0 - theta_grid * phi ** 2 * (1.0 + abs(rho)),
        ])

    def _fit_params(self, initial_guess: np.ndarray):
        """Least squares over the three global parameters, holding the theta curve fixed."""
        result = least_squares(self._residuals, initial_guess, bounds=self._bounds())
        params = result.x
        if self.enforce_butterfly and self.butterfly_margins(params).min() <= 0.0:
            params = self._constrained_fit(params)
        return params

    def _constrained_fit(self, initial_guess: np.ndarray):
        """Same least squares problem, but with butterfly as hard non-linear constraints.

        Only reached when the unconstrained optimum violates them: when it does not, it
        already solves the constrained problem and there is nothing to gain.
        """
        theta_grid = self._theta_grid()
        lower, upper = self._bounds()
        result = minimize(
            lambda p: 0.5 * float(self._residuals(p) @ self._residuals(p)),
            initial_guess,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=[{"type": "ineq", "fun": lambda p: self.butterfly_margins(p, theta_grid)}],
            options={"maxiter": 500, "ftol": 1e-14},
        )
        params = result.x
        # Both constrained quantities are increasing in eta, so shrinking eta always restores
        # feasibility if SLSQP stopped just outside the feasible set.
        for i in range(500):
            if self.butterfly_margins(params, theta_grid).min() > 0.0:
                return params
            params = np.array([params[0], params[1] * 0.98, params[2]])
        warnings.warn("Butterfly conditions could not be enforced; the fitted surface may admit butterfly arbitrage.")
        return params

    def _fit_theta(self, params: np.ndarray):
        """Refit each slice's ATM total variance with the global shape held fixed.

        Slices are visited in maturity order and each theta is bounded below by the previous
        one, so the calibrated term structure is non-decreasing by construction and the
        calendar condition never has to be repaired after the fact.
        """
        theta = np.empty(len(self.maturities))
        floor = 1e-8
        for i, (k, w_mkt, weights) in enumerate(zip(self.log_moneyness_by_maturity, self.market_total_variance_by_maturity, self.weights_by_maturity)):
            result = least_squares(
                lambda t: weights * (self.total_variance(params, t[0], k) - w_mkt),
                [max(self.theta[i], floor)], bounds=([floor], [10.0]),
            )
            theta[i] = result.x[0]
            floor = result.x[0]
        return theta

    def calibrate(self, initial_guess: np.ndarray = None) -> np.ndarray:
        """
        Alternate between the global parameters and the theta term structure.
        """
        x0 = self._initial_guess() if initial_guess is None else np.asarray(initial_guess, dtype=float)
        params = x0
        for i in range(self.max_iterations):
            params = self._fit_params(params)
            if not self.refit_theta:
                break
            theta = self._fit_theta(params)
            converged = np.max(np.abs(theta - self.theta)) < self.tol
            self.theta = theta
            if converged:
                break

        # The butterfly constraints were imposed on the theta range of the previous sweep,
        # and the final theta refit can move that range slightly. Re-check against the theta
        # that will actually be used, and repair if the surface has drifted outside.
        if self.enforce_butterfly and self.butterfly_margins(params).min() <= 0.0:
            params = self._constrained_fit(params)

        self.params = params
        self._build_theta_interpolator()
        return self.params

    def _build_theta_interpolator(self):
        """Interpolate theta between expiries without breaking monotonicity.

        PCHIP is used rather than a cubic spline exactly because it is shape preserving: a
        non-decreasing set of nodes gives a non-decreasing curve, so the calendar condition
        holds at every maturity and not only at the quoted ones. The (0, 0) node states that
        total variance vanishes as maturity goes to zero.
        """
        maturity_nodes = np.concatenate(([0.0], self.maturities))
        theta_nodes = np.concatenate(([0.0], self.theta))
        self._theta_interpolator = PchipInterpolator(maturity_nodes, theta_nodes, extrapolate=False)
        self._max_maturity = float(maturity_nodes[-1])
        self._max_theta = float(theta_nodes[-1])
        self._max_theta_slope = float(self._theta_interpolator.derivative()(self._max_maturity))

    def _theta_for_maturity(self, maturity: float) -> float:
        """ATM total variance at any maturity; extended linearly past the last expiry."""
        maturities = np.atleast_1d(np.asarray(maturity, dtype=float))
        theta = np.where(
            maturities <= self._max_maturity,
            self._theta_interpolator(np.clip(maturities, 0.0, self._max_maturity)),
            self._max_theta + max(self._max_theta_slope, 0.0) * (maturities - self._max_maturity),
        )
        theta = np.maximum(theta, 1e-12)
        return theta if np.ndim(maturity) else float(theta[0])

    def _forward_for_maturity(self, maturity: float) -> float:
        """Forwards are interpolated in log space, i.e. linearly in the carry rate."""
        return float(np.exp(np.interp(maturity, self.maturities, np.log(self.forwards))))

    def fitted_total_variance(self, k: np.ndarray, maturity: float) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Call calibrate() before requesting fitted values.")
        theta = self._theta_for_maturity(maturity)
        return self.total_variance(self.params, theta, np.asarray(k, dtype=float))

    def fitted_implied_vol(self, k: np.ndarray, maturity: float) -> np.ndarray:
        w = self.fitted_total_variance(k, maturity)
        return np.sqrt(w / maturity)

    def fitted_implied_vol_from_strike(self, strikes: np.ndarray, maturity: float, forward: float = None) -> np.ndarray:
        forward = self._forward_for_maturity(maturity) if forward is None else forward
        return self.fitted_implied_vol(np.log(np.asarray(strikes, dtype=float) / forward), maturity)

    def slice_derivatives(self, k: np.ndarray, maturity: float) -> tuple:
        """Analytic w, dw/dk and d2w/dk2 for one fitted slice."""
        if self.params is None:
            raise RuntimeError("Call calibrate() before requesting fitted values.")
        rho, eta, gamma = self.params
        theta = self._theta_for_maturity(maturity)
        phi = self.phi(theta, eta, gamma)
        k = np.asarray(k, dtype=float)
        u = phi * k + rho
        root = np.sqrt(u ** 2 + (1.0 - rho ** 2))
        w = theta / 2.0 * (1.0 + rho * phi * k + root)
        dw = theta * phi / 2.0 * (rho + u / root)
        d2w = theta * phi ** 2 / 2.0 * (1.0 - rho ** 2) / root ** 3
        return w, dw, d2w

    def butterfly_function(self, k: np.ndarray, maturity: float) -> np.ndarray:
        """Gatheral's g(k), whose positivity is equivalent to a non-negative density:
            g = (1 - k*w'/(2w))**2 - (w'**2/4)*(1/w + 1/4) + w''/2
        """
        w, dw, d2w = self.slice_derivatives(k, maturity)
        return (1.0 - k * dw / (2.0 * w)) ** 2 - (dw ** 2 / 4.0) * (1.0 / w + 0.25) + d2w / 2.0

    def implied_density(self, k: np.ndarray, maturity: float) -> np.ndarray:
        """Risk-neutral density of log-moneyness implied by the fitted slice."""
        w, _, _ = self.slice_derivatives(k, maturity)
        g = self.butterfly_function(k, maturity)
        d2 = -np.asarray(k, dtype=float) / np.sqrt(w) - np.sqrt(w) / 2.0
        return g / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d2 ** 2)

    def log_moneyness_range(self) -> tuple:
        all_k = np.concatenate(self.log_moneyness_by_maturity)
        return float(all_k.min()), float(all_k.max())

    def check_arbitrage(self, n_k: int = 201, n_maturities: int = 60, k_range: tuple = None) -> dict:
        """Verify both no-arbitrage conditions numerically, on a dense grid of the fitted surface.

        The constraints are imposed at calibration time on the theta nodes only, so this
        re-checks them where the surface is actually evaluated: between expiries and across
        the whole moneyness range.
        """
        if self.params is None:
            raise RuntimeError("Call calibrate() before checking arbitrage.")
        k_min, k_max = self.log_moneyness_range() if k_range is None else k_range
        k_grid = np.linspace(k_min, k_max, n_k)
        maturity_grid = np.linspace(self.maturities.min(), self.maturities.max(), n_maturities)

        total_variance = np.array([self.fitted_total_variance(k_grid, T) for T in maturity_grid])
        calendar_slack = float(np.min(np.diff(total_variance, axis=0)))
        butterfly_slack = float(min(self.butterfly_function(k_grid, T).min() for T in maturity_grid))

        return {
            "calendar_arbitrage_free": calendar_slack >= -1e-12,
            "min_total_variance_increment": calendar_slack,
            "theta_non_decreasing": bool(np.all(np.diff(self.theta) >= -1e-12)),
            "butterfly_arbitrage_free": butterfly_slack >= 0.0,
            "min_butterfly_g": butterfly_slack,
            "min_butterfly_margin": float(self.butterfly_margins(self.params).min()),
        }

    def fit_report(self) -> list[dict]:
        """Per-slice fit quality, expressed in volatility points."""
        if self.params is None:
            raise RuntimeError("Call calibrate() before requesting a fit report.")
        report = []
        for i, maturity in enumerate(self.maturities):
            k = self.log_moneyness_by_maturity[i]
            market_vol = self.market_implied_vol_by_maturity[i]
            fitted_vol = np.sqrt(self.total_variance(self.params, self.theta[i], k) / maturity)
            error = fitted_vol - market_vol
            report.append({
                "maturity": float(maturity),
                "n_quotes": len(k),
                "forward": float(self.forwards[i]),
                "theta": float(self.theta[i]),
                "atm_vol": float(np.sqrt(self.theta[i] / maturity)),
                "rmse_vol": float(np.sqrt(np.mean(error ** 2))),
                "max_abs_error_vol": float(np.max(np.abs(error))),
            })
        return report


if __name__ == "__main__":
    forward = 100.0
    maturity = 0.5
    strikes = np.array([70, 80, 90, 100, 110, 120, 130])
    implied_vols = np.array([0.32, 0.27, 0.24, 0.22, 0.23, 0.26, 0.30])

    calibrator = SVICalibrator(strikes, implied_vols, forward, maturity)
    params = calibrator.calibrate()
    a, b, rho, m, sigma = params

    print(f"Calibrated SVI parameters: a={a:.5f}, b={b:.5f}, rho={rho:.5f}, m={m:.5f}, sigma={sigma:.5f}")
    print("Strike | Market IV | Fitted IV")
    for strike, market_iv, fitted_iv in zip(strikes, implied_vols, calibrator.fitted_implied_vol()):
        print(f"{strike:6.1f} | {market_iv:9.4f} | {fitted_iv:9.4f}")

    print()

    maturities = [0.25, 0.5, 1.0]
    forwards = [100.0, 100.0, 100.0]
    strikes_by_maturity = [
        np.array([80, 90, 100, 110, 120]),
        np.array([70, 80, 90, 100, 110, 120, 130]),
        np.array([60, 80, 100, 120, 140]),
    ]
    implied_vols_by_maturity = [
        np.array([0.29, 0.25, 0.22, 0.24, 0.28]),
        np.array([0.32, 0.27, 0.24, 0.22, 0.23, 0.26, 0.30]),
        np.array([0.30, 0.24, 0.21, 0.23, 0.27]),
    ]

    ssvi_calibrator = SSVICalibrator(maturities, strikes_by_maturity, implied_vols_by_maturity, forwards)
    rho, eta, gamma = ssvi_calibrator.calibrate()

    print(f"Calibrated SSVI parameters: rho={rho:.5f}, eta={eta:.5f}, gamma={gamma:.5f}")
    for i, maturity in enumerate(ssvi_calibrator.maturities):
        print(f"\nMaturity T={maturity}")
        print("Strike | Market IV | Fitted IV")
        k = ssvi_calibrator.log_moneyness_by_maturity[i]
        fitted_ivs = ssvi_calibrator.fitted_implied_vol(k, maturity)
        for strike, market_iv, fitted_iv in zip(ssvi_calibrator.strikes_by_maturity[i],
                                                ssvi_calibrator.market_implied_vol_by_maturity[i], fitted_ivs):
            print(f"{strike:6.1f} | {market_iv:9.4f} | {fitted_iv:9.4f}")

    print("\nNo-arbitrage checks:")
    for name, value in ssvi_calibrator.check_arbitrage().items():
        print(f"  {name}: {value}")
