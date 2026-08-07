import numpy as np
from scipy.optimize import least_squares


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
    """

    def __init__(self, maturities, strikes_by_maturity, implied_vols_by_maturity, forwards):
        self.maturities = np.asarray(maturities, dtype=float)
        n_slices = len(self.maturities)
        if not (len(strikes_by_maturity) == n_slices and len(implied_vols_by_maturity) == n_slices and len(forwards) == n_slices):
            raise ValueError("maturities, strikes_by_maturity, implied_vols_by_maturity and forwards must have the same length.")
        self.forwards = np.asarray(forwards, dtype=float)

        self.log_moneyness_by_maturity = []
        self.market_total_variance_by_maturity = []
        self.theta = np.zeros(n_slices)

        for i in range(n_slices):
            strikes = np.asarray(strikes_by_maturity[i], dtype=float)
            implied_vols = np.asarray(implied_vols_by_maturity[i], dtype=float)
            k = np.log(strikes / self.forwards[i])
            w = implied_vols ** 2 * self.maturities[i]
            order = np.argsort(k)
            self.log_moneyness_by_maturity.append(k)
            self.market_total_variance_by_maturity.append(w)
            self.theta[i] = np.interp(0.0, k[order], w[order])

        self.params = None

    @staticmethod
    def phi(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
        return eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))

    @classmethod
    def total_variance(cls, params: np.ndarray, theta, k: np.ndarray) -> np.ndarray:
        rho, eta, gamma = params
        phi = cls.phi(theta, eta, gamma)
        return theta / 2.0 * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1.0 - rho ** 2)))

    def _residuals(self, params: np.ndarray) -> np.ndarray:
        residuals = [
            self.total_variance(params, theta, k) - w_mkt
            for k, w_mkt, theta in zip(self.log_moneyness_by_maturity, self.market_total_variance_by_maturity, self.theta)
        ]
        return np.concatenate(residuals)

    def _initial_guess(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.5])

    def _bounds(self) -> tuple:
        lower = [-0.999, 1e-6, 1e-6]
        upper = [0.999, 10.0, 1.0 - 1e-6]
        return lower, upper

    def calibrate(self, initial_guess: np.ndarray = None) -> np.ndarray:
        x0 = self._initial_guess() if initial_guess is None else np.asarray(initial_guess, dtype=float)
        result = least_squares(self._residuals, x0, bounds=self._bounds())
        self.params = result.x
        return self.params

    def _theta_for_maturity(self, maturity: float) -> float:
        order = np.argsort(self.maturities)
        return np.interp(maturity, self.maturities[order], self.theta[order])

    def fitted_implied_vol(self, k: np.ndarray, maturity: float) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Call calibrate() before requesting fitted implied vols.")
        theta = self._theta_for_maturity(maturity)
        w = self.total_variance(self.params, theta, np.asarray(k, dtype=float))
        return np.sqrt(w / maturity)


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
    for i, maturity in enumerate(maturities):
        print(f"\nMaturity T={maturity}")
        print("Strike | Market IV | Fitted IV")
        k = ssvi_calibrator.log_moneyness_by_maturity[i]
        fitted_ivs = ssvi_calibrator.fitted_implied_vol(k, maturity)
        for strike, market_iv, fitted_iv in zip(strikes_by_maturity[i], implied_vols_by_maturity[i], fitted_ivs):
            print(f"{strike:6.1f} | {market_iv:9.4f} | {fitted_iv:9.4f}")
