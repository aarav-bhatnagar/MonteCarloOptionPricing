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

def black_scholes_put(S0, K, r, sigma, T):
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
    CDF1 = norm.cdf(-d1)
    CDF2 = norm.cdf(-d2)
    discountedStrike = K * math.exp(-r * T)
    P = (discountedStrike * CDF2) - (S0 * CDF1)

    return P


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

def mc_european_call_antithetic(S0, K, r, sigma, T, n_paths, seed):
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
    if n_paths < 2 or n_paths % 2 == 1:
        raise ValueError("Number of paths must be at least 2 and even")

    M = n_paths // 2
    np.random.seed(seed)
    z_samples = np.random.normal(loc=0.0, scale=1.0, size=M)

    drift_adjustment = (r - ((sigma * sigma) / 2)) * T
    random_component = sigma * math.sqrt(T)
    simulated_prices_plus = list(map(lambda x: S0 * math.exp((random_component * x) + drift_adjustment), z_samples))
    simulated_prices_minus = list(map(lambda x: S0 * math.exp((random_component * -x) + drift_adjustment), z_samples))
    call_payoffs_plus = list(map(lambda x: max(x - K, 0), simulated_prices_plus))
    call_payoffs_minus = list(map(lambda x: max(x - K, 0), simulated_prices_minus))
    discounted_plus = list(map(lambda x: x * math.exp(-r * T), call_payoffs_plus))
    discounted_minus = list(map(lambda x: x * math.exp(-r * T), call_payoffs_minus))
    pair_average = list(map(lambda x, y: (x + y) / 2, discounted_plus, discounted_minus))
    option_price = sum(pair_average) / len(pair_average)
    std = float(np.std(pair_average, ddof=1))
    ste = std / math.sqrt(M)
    lb = option_price - 1.96 * ste
    ub = option_price + 1.96 * ste

    return {"option_price" : option_price, "ste" : ste, "lowerbound" : lb, "upperbound" : ub}

def mc_european_call_control(S0, K, r, sigma, T, n_paths, seed):
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
    X = list(map(lambda x: x * math.exp(-r * T), call_payoffs))
    Y = list(map(lambda x: x * math.exp(-r * T), simulated_prices))
    X_bar = sum(X) / len(X)
    Y_bar = sum(Y) / len(Y)
    beta = np.cov(X, Y, ddof=1)[0, 1] / np.var(Y, ddof=1)
    price_cv = X_bar - beta * (Y_bar - S0)
    adjusted = list(map(lambda x, y: x - beta * (y - S0), X, Y))
    std = float(np.std(adjusted, ddof=1))
    ste = std / math.sqrt(n_paths)
    lb = price_cv - 1.96 * ste
    ub = price_cv + 1.96 * ste

    return {"option_price" : price_cv, "beta" : beta, "ste" : ste, "lowerbound" : lb, "upperbound" : ub}

def mc_european_put(S0, K, r, sigma, T, n_paths, seed):
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
    put_payoffs = list(map(lambda x: max(K - x, 0), simulated_prices))
    discounted = list(map(lambda x: x * math.exp(-r * T), put_payoffs))
    option_price = sum(discounted) / len(discounted)
    std = float(np.std(discounted, ddof=1))
    ste = std / math.sqrt(n_paths)
    lb = option_price - 1.96 * ste
    ub = option_price + 1.96 * ste

    return {"option_price" : option_price, "ste" : ste, "lowerbound" : lb, "upperbound" : ub}

#Validation Test - put-call parity
def test(S0, K, r, sigma, T):
    expecteddifference = S0 - (K * math.exp(-r * T))
    difference = black_scholes_call(S0, K, r, sigma, T) - black_scholes_put(S0, K, r, sigma, T)
    return abs(expecteddifference - difference)

if __name__ == "__main__":
    for NPow in range(3, 4, 1):
        N = 10 ** NPow
        print("Convergence at n_paths = ", N)
        print("CALL OPTION") #Gives you the right but not the obligation to buy a stock at a set strike price at a given expiry date
        BS_call = black_scholes_call(100, 100, 0.05, 0.20, 1)
        MC_call = mc_european_call(100, 100, 0.05, 0.20, 1, N, 42)["option_price"]
        MC_A_call = mc_european_call_antithetic(100, 100, 0.05, 0.20, 1, N, 42)["option_price"]
        MC_C_call = mc_european_call_control(100, 100, 0.05, 0.20, 1, N, 42)["option_price"]
        print("Black scholes:", BS_call, "Monte carlo:", MC_call, "Monte carlo: antithetic variate:", MC_A_call, "Monte carlo: control variate:", MC_C_call)
        call_error = abs(BS_call - MC_call)
        print("Error", call_error)

        print("PUT OPTION") #Gives you the right but not the obligation to sell a stock at a set strike price at a given expiry date
        BS_put = black_scholes_put(100, 100, 0.05, 0.20, 1)
        MC_put = mc_european_put(100, 100, 0.05, 0.20, 1, N, 42)["option_price"]
        print("Black scholes:", BS_put, "Monte carlo:", MC_put)
        put_error = abs(BS_put - MC_put)
        print("Error", put_error)

        print("\n")

    #print(test(100, 100, 0.05, 0.20, 1))