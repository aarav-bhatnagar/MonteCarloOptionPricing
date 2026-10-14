import math
from scipy.stats import norm
import numpy as np

def black_scholes_call(S0, K, r, sigma, T):
    #Stock price, strike price, annual risk free interest rate, volatility, time
    if S0 < 0:
        raise ValueError("Stock price must be non negative")
    if K < 0:
        raise ValueError("Strike price must be non negative")
    if T < 0:
        raise ValueError("Time to expiry must be non negative")
    if T == 0 or sigma == 0:
        raise ValueError("Invalid Input")

    Denominator = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + ((r + ((sigma * sigma) / 2)) * T)) / Denominator
    d2 = d1 - Denominator
    CDF1 = norm.cdf(d1)
    CDF2 = norm.cdf(d2)
    discountedStrike = K * math.exp(-r * T)
    C = (S0 * CDF1) - (discountedStrike * CDF2)

    return C


def mc_european_call(S0, K, r, sigma, T, n_paths, seed):
    #n_paths = number of iterations
    #seed = seed for random number generator so you can reproduce results
    if S0 < 0:
        raise ValueError("Stock price must be non negative")
    if K < 0:
        raise ValueError("Strike price must be non negative")
    if T < 0:
        raise ValueError("Time to expiry must be non negative")
    if T == 0 or sigma == 0:
        raise ValueError("Invalid Input")
    if n_paths < 2:
        raise ValueError("Number of paths must be at least 2")

    np.random.seed(seed)
    z_samples = np.random.normal(loc=0.0, scale=1.0, size=n_paths)

    drift_adjustment = (r - ((sigma * sigma) / 2)) * T
    random_component = sigma * math.sqrt(T)
    simulated_prices = list(map(lambda x: S0 * math.exp((random_component * x) + drift_adjustment), z_samples))
    call_payoffs = list(map(lambda x: max(x - K, 0), simulated_prices))
    discounted = list(map(lambda x: x * math.exp(-r * T), call_payoffs))
    option_price = sum(discounted) / len(discounted)
    std = float(np.std(discounted, ddof=1))
    ste = std / math.sqrt(n_paths)
    lb = option_price - 1.96 * ste
    ub = option_price + 1.96 * ste

    return {"option_price" : option_price, "ste" : ste, "lowerbound" : lb, "upperbound" : ub}


if __name__ == "__main__":
    print(black_scholes_call(100, 100, 0.05, 0.20, 1))
    print(mc_european_call(100, 100, 0.05, 0.20, 1, 1000000, 1))