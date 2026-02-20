import numpy as np

class BinomialModel:

    def __init__(self, S0: float, T: float, n_periods: int, r: float, up: float):
        self.S0 = S0
        self.T = T
        self.n_periods = n_periods
        self.r = r
        self.up = up
        self.down = 1 / self.up
        self.delta_t = self.T / self.n_periods
        self.q = (np.exp(self.r * self.delta_t) - self.down) / (self.up - self.down)

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
                        raise ValueError("The option type must either european or american.")
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


binomial = BinomialModel(S0=100., T=0.25, n_periods=10, r=0.1194, up=1.03775)
security_lattice = binomial.create_security_price_lattice()
call_option_lattice = binomial.price_call_option(K=100, option_type='european')
put_option_lattice = binomial.price_put_option(K=100, option_type='american')