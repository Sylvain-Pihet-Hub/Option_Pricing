import numpy as np
from scipy.stats import norm
import yfinance as yf

class SecurityData:

    def __init__(self, ticker: str, period: str="1y"):
        self.ticker = ticker
        self.period = period
        self.prices = yf.Ticker(self.ticker).history(period=self.period)["Close"]
        self.current_price = self.prices.iloc[-1]

class BinomialModel:

    def __init__(self, S0: float, T: float, n_periods: int, r: float, c:float, sigma: float,):
        self.S0 = S0
        self.T = T
        self.n_periods = n_periods
        self.r = r
        self.c = c
        self.sigma = sigma
        self.delta_t = self.T / self.n_periods
        self.up = np.exp(self.sigma * np.sqrt(self.delta_t))
        self.down = 1 / self.up
        self.q = (np.exp((self.r - self.c) * self.delta_t) - self.down) / (self.up - self.down)

    def create_security_price_lattice(self):
        security_lattice = np.zeros((self.n_periods+1, self.n_periods+1))
        security_lattice[self.n_periods, 0] = self.S0
        for period in range(1, self.n_periods+1):
            for price in range(period+1):
                if price < period:
                    security_lattice[self.n_periods - price, period] = security_lattice[self.n_periods - price, period - 1] * self.down
                elif price == period:
                    security_lattice[self.n_periods - price, period] = security_lattice[self.n_periods - price + 1, period - 1] * self.up
                else:
                    break
        return security_lattice

    def price_call_option(self, K: float, option_type: str):
        payoff_lattice = np.zeros((self.n_periods + 1, self.n_periods + 1))
        security_price_lattice = self.create_security_price_lattice()
        payoff_lattice[:, self.n_periods] = np.maximum(security_price_lattice[:, self.n_periods] - K, 0)
        for period in range(self.n_periods-1, -1, -1):
            for price in range(period+1):
                if price <= period:
                    if option_type == "european":
                        payoff_lattice[self.n_periods - price, period] = (np.exp(-self.r * self.delta_t)) * (self.q * payoff_lattice[self.n_periods - price - 1, period + 1] +
                                                                                     (1 - self.q) * payoff_lattice[self.n_periods - price, period + 1])
                    elif option_type == "american":
                        payoff_lattice[self.n_periods - price, period] = max((np.exp(-self.r * self.delta_t)) * (self.q * payoff_lattice[self.n_periods - price - 1, period + 1] +
                                                                                                             (1 - self.q) * payoff_lattice[self.n_periods - price, period + 1]), security_price_lattice[self.n_periods - price, period] - K)
                    else:
                        raise ValueError("The option type must be either european or american.")
        return payoff_lattice

    def price_put_option(self, K: float, option_type: str):
        payoff_lattice = np.zeros((self.n_periods + 1, self.n_periods + 1))
        security_price_lattice = self.create_security_price_lattice()
        payoff_lattice[:, self.n_periods] = np.maximum(K - security_price_lattice[:, self.n_periods], 0)
        for period in range(self.n_periods-1, -1, -1):
            for price in range(period+1):
                if price <= period:
                    if option_type == "european":
                        payoff_lattice[self.n_periods - price, period] = (np.exp(-self.r * self.delta_t)) * (self.q * payoff_lattice[self.n_periods - price - 1, period + 1] +
                                                                                     (1 - self.q) * payoff_lattice[self.n_periods - price, period + 1])
                    elif option_type == "american":
                        payoff_lattice[self.n_periods - price, period] = max((np.exp(-self.r * self.delta_t)) * (self.q * payoff_lattice[self.n_periods - price - 1, period + 1] +
                                                                                                             (1 - self.q) * payoff_lattice[self.n_periods - price, period + 1]), K - security_price_lattice[self.n_periods - price, period])
                    else:
                        raise ValueError("The option type must either european or american.")
        return payoff_lattice

class MonteCarloModel:

    def __init__(self, S0: float, T: float, n_periods: int, r: float, c: float, sigma: float, seed: int=120):
        self.S0 = S0
        self.T = T
        self.n_periods = n_periods
        self.r = r
        self.c = c
        self.sigma = sigma
        self.delta_t = self.T / self.n_periods
        self.seed = seed

    def create_security_price_paths(self, n_simulations: int=10_000):
        security_paths = np.zeros((n_simulations, self.n_periods + 1))
        security_paths[:, 0] = self.S0
        self.rng = np.random.default_rng(self.seed)
        for period in range(1, self.n_periods+1):
            z = self.rng.normal(loc=0, scale=1, size=n_simulations)
            security_paths[: , period] = security_paths[: , period - 1] * np.exp((self.r - self.c - 0.5 * self.sigma**2) * self.delta_t + self.sigma * np.sqrt(self.delta_t) * z)
        return security_paths

    def price_call_option(self, K: float):
        security_price_paths = self.create_security_price_paths()
        payoff = np.maximum(security_price_paths[:, -1] - K, 0)
        return np.mean(payoff)

    def price_put_option(self, K: float):
        security_price_paths = self.create_security_price_paths()
        payoff = np.maximum(K - security_price_paths[:, -1], 0)
        return np.mean(payoff)

class BlackScholesModel:

    def __init__(self, S0: float, T: float, K: float, r: float, c: float, sigma: float):
        self.S0 = S0
        self.T = T
        self.K = K
        self.r = r
        self.c = c
        self.sigma = sigma
        self.d1 = (np.log(self.S0 / self.K) + (self.r - self.c + self.sigma**2/2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def price_call_option(self):
        c0 = np.exp(-self.c * self.T) * self.S0 * norm.cdf(self.d1) - np.exp(-self.r * self.T) * self.K * norm.cdf(self.d2)
        return c0

    def price_put_option(self):
        c0 = self.price_call_option()
        p0 = c0 + self.K * np.exp(-self.r * self.T) - self.S0 * np.exp(-self.c * self.T)
        return p0