import numpy as np
from scipy.stats import norm

from option_pricing import OptionChainData

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

    def __init__(self, market_price: float, S0: float, rate: float, div_yield: float, option_data: EuropeanOption, max_iter: int=200, tol: float=1e-6):
        self.market_price = market_price
        self.S0 = S0
        self.rate = rate
        self.div_yield = div_yield
        self.option_data = option_data
        self.max_iter = max_iter
        self.tol = tol

    def rootfinder(self):
        implied_vol = 0.05
        low = 10**(-4)
        high = 2.0

        def bs_price(vol: float):
            market_data = MarketData(S0=self.S0, rate=self.rate, div_yield=self.div_yield, vol=vol)
            bs = BlackScholes(option_data=self.option_data, market_data=market_data)
            return bs.pricer_engine()

        if bs_price(low) - self.market_price > 0 or bs_price(high) - self.market_price < 0:
            raise ValueError('Implied volatility cannot be computed')

        # Newton method first as faster convergence
        for i in range(self.max_iter):
            market_data = MarketData(S0=self.S0, rate=self.rate, div_yield=self.div_yield, vol=implied_vol)
            bs = BlackScholes(option_data=self.option_data, market_data=market_data)
            model_price = bs.pricer_engine()
            if abs(model_price - self.market_price) < self.tol:
                return implied_vol
            vega = bs.vega()
            if vega < 10 ** (-4):
                break
            implied_vol -= (model_price - self.market_price) / vega

        # Secant bisection method
        for i in range(self.max_iter):
            mid = (low + high) / 2
            if abs(bs_price(mid) - self.market_price) < self.tol:
                return mid
            elif bs_price(mid) < self.market_price:
                low = mid
            else:
                high = mid
        return ValueError('No convergence for Implied Volatility')

def solve_implied_vols(chain_data: OptionChainData, expiry: str, rate: float, div_yield: float, is_call: bool=True, min_vol: float=0.001, max_vol: float=2.3):
    results = []
    for point in chain_data.market_datapoints(expiry, is_call):
        option_data = EuropeanOption(K=point['K'], T=point['T'], is_call=is_call)
        solver = ImpliedVol(market_price=point['market_price'], S0=chain_data.spot, rate=rate, div_yield=div_yield, option_data=option_data)
        try:
            implied_vol = solver.rootfinder()
        except ValueError:
            continue
        if not (min_vol <= implied_vol <= max_vol):
            continue
        results.append({**point, 'implied_vol': implied_vol})
    return results

def implied_vol_surface(ticker: str, is_call: bool=True, min_vol: float=0.001, max_vol: float=2.3, max_maturities: int=None):
    option_data = OptionChainData(ticker)
    rate = option_data.risk_free_rate()
    div_yield = option_data.dividend_yield()
    expirations = [expiry for expiry in option_data.expirations if option_data.time_to_maturity(expiry) > 0]
    if max_maturities is not None:
        expirations = expirations[:max_maturities]

    surface = []
    for expiry in expirations:
        surface.extend(solve_implied_vols(option_data, expiry, rate, div_yield, is_call, min_vol, max_vol))
    return surface

if __name__ == "__main__":
    ticker = 'AMD'
    surface = implied_vol_surface(ticker, is_call=True, max_maturities=15)

    print(f"{ticker} implied volatility surface (calls)")
    print("Expiry Date   | Maturity   | Strike | Implied Vol | Yahoo IV")
    for point in surface:
        print(f"{point['expiry']} | {point['T']:10.4f} | {point['K']:6.2f} | {point['implied_vol']:11.4f} | {point['yahoo_implied_vol']:8.4f}")


    # option_data = OptionChainData(ticker)
    # option_data.security.options
    # target_option = option_data.security.option_chain(date='2026-08-14').calls[['lastTradeDate', 'strike', 'lastPrice', 'impliedVolatility']]
    # target_option[target_option['strike'] == 1040]
