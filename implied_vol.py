import warnings
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

from option_pricing import OptionChainData
from svi_calibration import SSVICalibrator

class MarketData:
    def __init__(self, S0: float, rate: float, div_yield: float, vol: float):
        self.S0 = S0
        self.rate = rate
        self.div_yield = div_yield
        self.vol = vol

class EuropeanOption:
    def __init__(self, K: float, T: float, is_call: bool):
        self.K = K
        self.T = T
        self.is_call = is_call

class BlackScholes:
    def __init__(self, option_data: EuropeanOption, market_data: MarketData):
        self.option_data = option_data
        self.market_data = market_data

    def pricer_engine(self):
        d1 = ((np.log(self.market_data.S0 / self.option_data.K) + (self.market_data.rate - self.market_data.div_yield + 0.5 * self.market_data.vol**2) * self.option_data.T) /
              (self.market_data.vol * np.sqrt(self.option_data.T)))
        d2 = d1 - self.market_data.vol * np.sqrt(self.option_data.T)
        c0 = self.market_data.S0 * np.exp(-self.market_data.div_yield * self.option_data.T) * norm.cdf(d1) - self.option_data.K * np.exp(-self.market_data.rate * self.option_data.T) * norm.cdf(d2)
        if self.option_data.is_call:
            return c0
        else:
            p0 = c0 + self.option_data.K * np.exp(-self.market_data.rate * self.option_data.T) - self.market_data.S0 * np.exp(-self.market_data.div_yield * self.option_data.T)
            return p0

    def vega(self):
        d1 = ((np.log(self.market_data.S0 / self.option_data.K) + (self.market_data.rate - self.market_data.div_yield + 0.5 * self.market_data.vol ** 2) * self.option_data.T) /
              (self.market_data.vol * np.sqrt(self.option_data.T)))
        return self.market_data.S0 * np.exp(-self.market_data.div_yield * self.option_data.T) * norm.pdf(d1) * np.sqrt(self.option_data.T)

class ImpliedVol:

    def __init__(self, market_price: float, S0: float, rate: float, div_yield: float, option_data: EuropeanOption,
                 max_iter: int=100, price_tol: float=None, vol_tol: float=1e-8, min_vega_frac: float=1e-6):
        self.market_price = market_price
        self.S0 = S0
        self.rate = rate
        self.div_yield = div_yield
        self.option_data = option_data
        self.max_iter = max_iter
        # Price tolerance must scale with the instrument: 1e-6 absolute is unreachable precision
        # on a $20,000 index and meaningless inside a $0.50-wide market.
        self.price_tol = price_tol if price_tol is not None else 1e-8 * S0
        self.vol_tol = vol_tol
        self.min_vega_frac = min_vega_frac

    def initial_guess(self, low: float, high: float):
        """
        Brenner-Subrahmanyam seed: for an at-the-money option C ~ 0.4 * S0 * vol * sqrt(T),
        so vol ~ sqrt(2*pi/T) * C/S0. Far closer to typical equity vols than a fixed 0.05,
        so Newton converges in fewer steps and is less likely to step outside the bracket.
        """
        seed = np.sqrt(2 * np.pi / self.option_data.T) * self.market_price / self.S0
        return float(np.clip(seed, low, high))

    def rootfinder(self):
        low = 10**(-4)
        high = 2.0

        def bs_price(vol: float):
            market_data = MarketData(S0=self.S0, rate=self.rate, div_yield=self.div_yield, vol=vol)
            bs = BlackScholes(option_data=self.option_data, market_data=market_data)
            return bs.pricer_engine()

        if bs_price(low) - self.market_price > 0 or bs_price(high) - self.market_price < 0:
            raise ValueError('Implied volatility cannot be computed')

        implied_vol = self.initial_guess(low, high)

        # Newton method first as faster convergence
        # Vega is compared against S0 * sqrt(T), its natural scale, rather than an absolute
        # threshold: a bare 1e-4 means something completely different for a $10 stock than
        min_vega = self.min_vega_frac * self.S0 * np.sqrt(self.option_data.T)
        for i in range(self.max_iter):
            market_data = MarketData(S0=self.S0, rate=self.rate, div_yield=self.div_yield, vol=implied_vol)
            bs = BlackScholes(option_data=self.option_data, market_data=market_data)
            model_price = bs.pricer_engine()
            if abs(model_price - self.market_price) < self.price_tol:
                return implied_vol
            vega = bs.vega()
            if vega < min_vega:
                break
            implied_vol -= (model_price - self.market_price) / vega
            if not low <= implied_vol <= high:
                break
            if abs((model_price - self.market_price) / vega) < self.vol_tol:
                return implied_vol

        # Bisection fallback, guaranteed to converge inside the bracket checked above
        for i in range(self.max_iter):
            mid = (low + high) / 2
            price_gap = bs_price(mid) - self.market_price
            if abs(price_gap) < self.price_tol or (high - low) < self.vol_tol:
                return mid
            elif price_gap < 0:
                low = mid
            else:
                high = mid
        raise ValueError('No convergence for Implied Volatility')

class IVSurface:
    def __init__(self, ticker: str, min_vol: float=0.001, max_vol: float=2.0, max_maturity: float=None, max_rel_spread: float=0.20, min_open_interest: int=5, min_volume: int=0):
        self.ticker = ticker
        self.min_vol = min_vol
        self.max_vol = max_vol
        self.max_maturity = max_maturity
        self.max_rel_spread = max_rel_spread
        self.min_open_interest = min_open_interest
        self.min_volume = min_volume

    def check_data_quality(self, point: dict):
        """
        Drop low-information quotes before inversion. Wide or illiquid markets carry little
        volatility signal, and because implied vol is recovered by inverting the price, that
        quote noise is amplified straight into the surface.
        """
        bid, ask = point['bid'], point['ask']
        if bid <= 0 or ask <= 0 or ask < bid:
            return False

        mid = (bid + ask) / 2
        if (ask - bid) / mid > self.max_rel_spread:
            return False

        volume = 0 if point['volume'] is None or np.isnan(point['volume']) else point['volume']
        open_interest = 0 if point['open_interest'] is None or np.isnan(point['open_interest']) else point['open_interest']
        if open_interest < self.min_open_interest or volume < self.min_volume:
            return False
        return True

    def solve_implied_vols(self, chain_data: OptionChainData, expiry: str, rate: float, div_yield: float):
        results = []
        for point in chain_data.otm_datapoints(expiry):
            if not self.check_data_quality(point):
                continue
            option_data = EuropeanOption(K=point['K'], T=point['T'], is_call=point['is_call'])
            solver = ImpliedVol(market_price=point['mid'], S0=chain_data.spot, rate=rate, div_yield=div_yield, option_data=option_data)
            try:
                implied_vol = solver.rootfinder()
            except ValueError:
                continue
            if not (self.min_vol <= implied_vol <= self.max_vol):
                continue

            # SSVI lives in (log-moneyness against the forward, total variance) coordinates, and
            # weights each quote by its price sensitivity, so both are attached here while the
            # pricing inputs that produced the vol are still in scope.
            market_data = MarketData(S0=chain_data.spot, rate=rate, div_yield=div_yield, vol=implied_vol)
            vega = BlackScholes(option_data=option_data, market_data=market_data).vega()
            forward = chain_data.spot * np.exp((rate - div_yield) * point['T'])
            results.append({
                **point,
                'implied_vol': implied_vol,
                'spot': chain_data.spot,
                'forward': forward,
                'moneyness': point['K'] / forward,
                'log_moneyness': float(np.log(point['K'] / forward)),
                'total_variance': implied_vol ** 2 * point['T'],
                'vega': vega,
            })
        return results

    def implied_vol_surface(self):
        option_data = OptionChainData(self.ticker)
        self.S0 = option_data.spot
        self.rate = rate = option_data.risk_free_rate()
        self.div_yield = div_yield = option_data.dividend_yield()
        expirations = [expiry for expiry in option_data.expirations if option_data.time_to_maturity(expiry) > 0]
        if self.max_maturity is not None:
            expirations = [expiry for expiry in expirations if option_data.time_to_maturity(expiry) < self.max_maturity]

        surface = []
        for expiry in expirations:
            surface.extend(self.solve_implied_vols(option_data, expiry, rate, div_yield))

        if not surface:
            # Implied vol is inverted from the bid/ask midpoint, and most feeds report bid=ask=0
            # once the session closes, so every quote fails the two-sided market check.
            warnings.warn(f"No usable quotes for {self.ticker}: every contract was filtered out. "
                          f"Outside trading hours bid/ask are typically reported as 0, which no "
                          f"amount of loosening max_rel_sp"
                          f"read or min_open_interest will fix.")
        return surface

class SSVISurfacePlot:
    """
    Turns the discrete implied vol grid produced by IVSurface into a smooth, arbitrage-free
    surface and draws it.
    """

    PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, points: list[dict], min_points_per_slice: int = 5, moneyness: str = 'log', **calibrator_kwargs):
        self.points = points
        self.min_points_per_slice = min_points_per_slice
        self.moneyness = moneyness
        self.expiries, maturities, strikes, implied_vols, forwards, weights = self.group_by_expiry(points, min_points_per_slice)

        if len(maturities) < 2:
            raise ValueError(f"SSVI needs at least 2 expiries with {min_points_per_slice}+ usable "
                             f"quotes each; got {len(maturities)}.")
        self.calibrator = SSVICalibrator(maturities, strikes, implied_vols, forwards, weights_by_maturity=weights, **calibrator_kwargs)
        self.params = self.calibrator.calibrate()

    @classmethod
    def from_ticker(cls, ticker: str, max_maturity: float = None, min_points_per_slice: int = 5,
                    moneyness: str = 'log', surface_kwargs: dict = None, **calibrator_kwargs):
        """End to end: fetch the chain, invert every usable quote, calibrate and plot."""
        iv_surface = IVSurface(ticker, max_maturity=max_maturity, **(surface_kwargs or {}))
        points = iv_surface.implied_vol_surface()
        if not points:
            raise ValueError(f"No usable implied vol points for {ticker}.")
        plot = cls(points, min_points_per_slice=min_points_per_slice, moneyness=moneyness, **calibrator_kwargs)
        plot.ticker = ticker
        return plot

    @staticmethod
    def group_by_expiry(points: list[dict], min_points_per_slice: int = 5) -> tuple:
        """
        Reshape the flat list of implied vol points into the per-maturity arrays
        SSVICalibrator expects: expiries, maturities, strikes, implied vols, forwards, weights.

        Residuals are measured in total variance, so a quote is weighted by dPrice/dw =
        vega / (2*sigma*T), which converts a variance error into the price error it causes.
        That anchors the fit where the market carries information - around the money - rather
        than letting the wings, where vega is negligible and spreads widest, drive the surface.
        """
        grouped = defaultdict(list)
        for point in points:
            grouped[point['expiry']].append(point)

        expiries, maturities, strikes, implied_vols, forwards, weights = [], [], [], [], [], []
        for expiry in sorted(grouped, key=lambda e: grouped[e][0]['T']):
            quotes = grouped[expiry]
            if len(quotes) < min_points_per_slice:
                continue
            T = quotes[0]['T']
            vol = np.array([q['implied_vol'] for q in quotes], dtype=float)
            vega = np.array([q['vega'] for q in quotes], dtype=float)
            expiries.append(expiry)
            maturities.append(T)
            strikes.append(np.array([q['K'] for q in quotes], dtype=float))
            implied_vols.append(vol)
            forwards.append(quotes[0]['forward'])
            weights.append(vega / (2.0 * vol * T))

        return expiries, np.array(maturities), strikes, implied_vols, np.array(forwards), weights

    def _axis(self, k, moneyness: str = None) -> np.ndarray:
        """Map log-moneyness onto the requested moneyness convention."""
        moneyness = self.moneyness if moneyness is None else moneyness
        if moneyness == 'log':
            return np.asarray(k, dtype=float)
        if moneyness == 'forward':
            return np.exp(np.asarray(k, dtype=float))
        raise ValueError("moneyness must be 'log' for log(K/F) or 'forward' for K/F.")

    def _axis_label(self, moneyness: str = None) -> str:
        moneyness = self.moneyness if moneyness is None else moneyness
        return 'log-moneyness log(K/F)' if moneyness == 'log' else 'moneyness K/F'

    def surface_grid(self, n_k: int = 90, n_maturities: int = 60, k_range: tuple = None,
                     maturity_range: tuple = None) -> tuple:
        """Dense (log-moneyness, maturity, implied vol) mesh of the calibrated surface."""
        calibrator = self.calibrator
        k_min, k_max = calibrator.log_moneyness_range() if k_range is None else k_range
        if maturity_range is None:
            maturity_range = (calibrator.maturities.min(), calibrator.maturities.max())
        k_grid = np.linspace(k_min, k_max, n_k)
        maturity_grid = np.linspace(*maturity_range, n_maturities)
        vol_grid = np.array([calibrator.fitted_implied_vol(k_grid, T) for T in maturity_grid])
        return k_grid, maturity_grid, vol_grid

    def plot_surface(self, n_k: int = 90, n_maturities: int = 60, show_market: bool = True,
                     moneyness: str = None, title: str = None) -> go.Figure:
        """The calibrated SSVI surface, with the market implied vols overlaid as points."""
        calibrator = self.calibrator
        k_grid, maturity_grid, vol_grid = self.surface_grid(n_k, n_maturities)

        figure = go.Figure(go.Surface(x=self._axis(k_grid, moneyness), y=maturity_grid, z=vol_grid,
                                      colorscale='Viridis', opacity=0.9, name='SSVI',
                                      colorbar=dict(title='implied vol')))
        if show_market:
            market_maturity = np.concatenate([np.full(len(k), T) for k, T in
                                              zip(calibrator.log_moneyness_by_maturity, calibrator.maturities)])
            figure.add_trace(go.Scatter3d(
                x=self._axis(np.concatenate(calibrator.log_moneyness_by_maturity), moneyness),
                y=market_maturity,
                z=np.concatenate(calibrator.market_implied_vol_by_maturity),
                mode='markers', name='market', marker=dict(size=3, color='crimson'),
            ))

        rho, eta, gamma = self.params
        figure.update_layout(
            title=title or f"{getattr(self, 'ticker', 'SSVI')} implied volatility surface "
                           f"(rho={rho:.3f}, eta={eta:.3f}, gamma={gamma:.3f})",
            scene=dict(xaxis_title=self._axis_label(moneyness), yaxis_title='maturity (years)',
                       zaxis_title='implied volatility'),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return figure

    def plot_smiles(self, n_k: int = 200, moneyness: str = None) -> go.Figure:
        """Market against fitted smile, one curve per expiry: the 2-D view of the same fit."""
        calibrator = self.calibrator
        figure = go.Figure()
        for i, (k, market_vol, T) in enumerate(zip(calibrator.log_moneyness_by_maturity,
                                                   calibrator.market_implied_vol_by_maturity,
                                                   calibrator.maturities)):
            color = self.PALETTE[i % len(self.PALETTE)]
            k_grid = np.linspace(k.min(), k.max(), n_k)
            figure.add_trace(go.Scatter(x=self._axis(k_grid, moneyness),
                                        y=calibrator.fitted_implied_vol(k_grid, T),
                                        mode='lines', line=dict(color=color),
                                        name=f"{self.expiries[i]} fit"))
            figure.add_trace(go.Scatter(x=self._axis(k, moneyness), y=market_vol, mode='markers',
                                        marker=dict(color=color, size=6, symbol='circle-open'),
                                        name=f"{self.expiries[i]} market", showlegend=False))

        figure.update_layout(title='SSVI smiles by expiry', xaxis_title=self._axis_label(moneyness),
                             yaxis_title='implied volatility')
        return figure

    def plot_term_structure(self, n_maturities: int = 200) -> go.Figure:
        """
        ATM total variance against maturity. This is the picture of the calendar condition:
        the interpolated curve must never turn down, or a calendar spread would be free money.
        """
        calibrator = self.calibrator
        maturity_grid = np.linspace(1e-4, calibrator.maturities.max(), n_maturities)
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=maturity_grid,
                                    y=[calibrator._theta_for_maturity(T) for T in maturity_grid],
                                    mode='lines', name='fitted theta(T)'))
        figure.add_trace(go.Scatter(x=calibrator.maturities, y=calibrator.theta, mode='markers',
                                    marker=dict(size=8, color='crimson'), name='calibrated nodes'))
        figure.update_layout(title='ATM total variance term structure (calendar condition)',
                             xaxis_title='maturity (years)', yaxis_title='theta = ATM total variance')
        return figure

    def plot_density(self, maturities: list = None, n_k: int = 400, k_range: tuple = None) -> go.Figure:
        """
        Risk-neutral density implied by the fitted slices. This is the picture of the butterfly
        condition: the density dipping below zero is exactly what butterfly arbitrage means.
        """
        calibrator = self.calibrator
        maturities = list(calibrator.maturities) if maturities is None else maturities
        k_min, k_max = calibrator.log_moneyness_range() if k_range is None else k_range
        k_grid = np.linspace(k_min, k_max, n_k)

        figure = go.Figure()
        for i, T in enumerate(maturities):
            figure.add_trace(go.Scatter(x=k_grid, y=calibrator.implied_density(k_grid, T),
                                        mode='lines', line=dict(color=self.PALETTE[i % len(self.PALETTE)]),
                                        name=f"T={T:.3f}"))
        figure.add_hline(y=0.0, line=dict(color='black', width=1, dash='dot'))
        figure.update_layout(title='Implied risk-neutral density (butterfly condition)',
                             xaxis_title='log-moneyness log(K/F)', yaxis_title='density')
        return figure

    def summary(self) -> str:
        """Calibrated parameters, per-slice fit quality and the two no-arbitrage checks."""
        rho, eta, gamma = self.params
        lines = [f"{len(self.points)} implied vols across {len(self.calibrator.maturities)} expiries",
                 f"SSVI parameters: rho={rho:.4f}, eta={eta:.4f}, gamma={gamma:.4f}",
                 "",
                 f"{'expiry':>12} {'T':>7} {'n':>4} {'ATM vol':>9} {'RMSE':>8} {'max err':>8}"]
        for expiry, row in zip(self.expiries, self.calibrator.fit_report()):
            lines.append(f"{expiry:>12} {row['maturity']:7.3f} {row['n_quotes']:4d} "
                         f"{row['atm_vol']:9.4f} {row['rmse_vol']:8.4f} {row['max_abs_error_vol']:8.4f}")
        lines.append("")
        lines.append("No-arbitrage checks:")
        for name, value in self.calibrator.check_arbitrage().items():
            lines.append(f"  {name}: {value}")
        return "\n".join(lines)


if __name__ == "__main__":
    surface = SSVISurfacePlot.from_ticker("PLTR", max_maturity=0.25)
    print(surface.summary())
    surface.plot_surface().show()
    surface.plot_smiles().show()
    surface.plot_term_structure().show()
    surface.plot_density().show()

