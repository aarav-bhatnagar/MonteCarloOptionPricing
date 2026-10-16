import math
import numpy as np
from scipy.stats import norm

def mc_asian_call(S0, K, r, sigma, T, n_paths, n_steps, seed):
    #n_paths = number of iterations
    #seed = seed for random number generator so you can reproduce results
    #Current stock price, strike price, risk free rate, volatility, time, iterations, steps, seed
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

    dt = T / n_steps
    drift_adjustment = (r - ((sigma * sigma) / 2)) * dt
    diffusion = sigma * math.sqrt(dt)

    discounted_payoffs = [0] * n_paths
    for path in range(n_paths):
        stock_price = S0
        total = 0

        for step in range(n_steps):
            Z = np.random.normal(loc=0.0, scale=1.0)

            stock_price = stock_price * math.exp(drift_adjustment + diffusion * Z)

            total += stock_price

        average_price = total / n_steps
        call_payoff = max(average_price - K, 0)
        discounted_payoff = call_payoff * math.exp(-r * T)

        discounted_payoffs[path] = discounted_payoff

    option_price = sum(discounted_payoffs) / n_paths
    std = float(np.std(discounted_payoffs, ddof=1))
    ste = std / math.sqrt(n_paths)
    lb = option_price - 1.96 * ste
    ub = option_price + 1.96 * ste

    return {"option_price" : option_price, "ste" : ste, "lowerbound" : lb, "upperbound" : ub}

def mc_asian_put(S0, K, r, sigma, T, n_paths, n_steps, seed):
    #n_paths = number of iterations
    #seed = seed for random number generator so you can reproduce results
    #Current stock price, strike price, risk free rate, volatility, time, iterations, steps, seed
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

    dt = T / n_steps
    drift_adjustment = (r - ((sigma * sigma) / 2)) * dt
    diffusion = sigma * math.sqrt(dt)

    discounted_payoffs = [0] * n_paths
    for path in range(n_paths):
        stock_price = S0
        total = 0

        for step in range(n_steps):
            Z = np.random.normal(loc=0.0, scale=1.0)

            stock_price = stock_price * math.exp(drift_adjustment + diffusion * Z)

            total += stock_price

        average_price = total / n_steps
        put_payoff = max(K - average_price, 0)
        discounted_payoff = put_payoff * math.exp(-r * T)

        discounted_payoffs[path] = discounted_payoff

    option_price = sum(discounted_payoffs) / n_paths
    std = float(np.std(discounted_payoffs, ddof=1))
    ste = std / math.sqrt(n_paths)
    lb = option_price - 1.96 * ste
    ub = option_price + 1.96 * ste

    return {"option_price" : option_price, "ste" : ste, "lowerbound" : lb, "upperbound" : ub}

def geometric_asian_price(S0, K, r, sigma, T, n_steps):
    #Analytical price for a discretely monitored geometric Asian option.

    mu = math.log(S0) + (r - sigma**2 / 2) * T * (
        n_steps + 1
    ) / (2 * n_steps)

    variance = sigma**2 * T * (n_steps + 1) * (
        2 * n_steps + 1
    ) / (6 * n_steps**2)

    sqrt_variance = math.sqrt(variance)

    d2 = (mu - math.log(K)) / sqrt_variance
    d1 = d2 + sqrt_variance

    price = math.exp(-r * T) * (
        math.exp(mu + variance / 2) * norm.cdf(d1)
        - K * norm.cdf(d2)
    )

    return price

def mc_asian_call_cv(S0, K, r, sigma, T, n_paths, n_steps, seed):
    #n_paths = number of iterations
    #seed = seed for random number generator so you can reproduce results
    #Current stock price, strike price, risk free rate, volatility, time, iterations, steps, seed
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

    dt = T / n_steps
    drift_adjustment = (r - ((sigma * sigma) / 2)) * dt
    diffusion = sigma * math.sqrt(dt)

    X = [0] * n_paths #ARITHMETIC
    Y = [0] * n_paths #GEOMETRIC
    for path in range(n_paths):
        stock_price = S0
        total_sum = 0
        logarithmic_sum = 0
        total_product = None

        for step in range(n_steps):
            Z = np.random.normal(loc=0.0, scale=1.0)

            stock_price = stock_price * math.exp(drift_adjustment + diffusion * Z)

            total_sum += stock_price
            logarithmic_sum += math.log(stock_price)
            #total_product = total_product * stock_price if total_product else stock_price

        arithmetic_average_price = total_sum / n_steps

        logarithmic_average = logarithmic_sum / n_steps
        geometric_average_price = math.exp(logarithmic_average)
        #geometric_average_price = total_product ** (1 / n_steps)

        X[path] = math.exp(-r * T) * max(arithmetic_average_price - K, 0)
        Y[path] = math.exp(-r * T) * max(geometric_average_price - K, 0)

    X_bar = sum(X) / n_paths
    Y_bar = sum(Y) / n_paths

    beta = np.cov(X, Y, ddof=1)[0, 1] / np.var(Y, ddof=1)

    CG = geometric_asian_price(S0, K,r, sigma, T, n_steps)
    option_price = X_bar - (beta * (Y_bar - CG))
    adjusted = list(map(lambda x, y: x - beta * (y - CG), X, Y))
    std = float(np.std(adjusted, ddof=1))
    ste = std / math.sqrt(n_paths)
    lb = option_price - 1.96 * ste
    ub = option_price + 1.96 * ste

    return {"option_price" : option_price, "beta" : beta, "ste" : ste, "lowerbound" : lb, "upperbound" : ub}

if __name__ == "__main__":
    params = (100, 100, 0.05, 0.20, 1, 100000, 100, 42)

    print("Plain Asian call:")
    print(mc_asian_call(*params))

    print("Control-variate Asian call:")
    print(mc_asian_call_cv(*params))


    """
    print("(n_paths, call option price, call ste, put option price, put ste)")
    for NPow in range(3, 6, 1):
        line = ""
        N = 10 ** NPow

        line += str(N) + " "
        call_data = mc_asian_call(100, 100, 0.05, 0.20, 1, N, 100, 42)
        line += str(call_data["option_price"]) + " " + str(call_data["ste"]) + " "

        put_data = mc_asian_put(100, 100, 0.05, 0.20, 1, N, 100, 42)
        line += str(put_data["option_price"]) + " " + str(put_data["ste"]) + " "

        print(line)
    """