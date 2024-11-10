import numpy as np
import pandas as pd

# Poisson Jumps
def jumps(lambda_, T, gamma, delta):
    t = 0
    jumps = 0
    jump_values = [0]
    jump_times = [0]
    while t < T:
        t = t + np.random.exponential(1 / lambda_)
        jumps = jumps + np.random.normal(gamma, delta)
        if t < T:
            jump_values.append(jumps)
            jump_times.append(t)
    return jump_values, jump_times


#2D Merton Jump Diffusion Process
def MJD_2d(S1, S2, T, n, lambda_, theta1, theta2, rho):
    # Parameter fetching
    mu1, sig1, gamma1, delta1 = theta1
    mu2, sig2, gamma2, delta2 = theta2

    dt = T / n
    price1 = np.zeros(n+1)
    price2 = np.zeros(n+1)
    price1[0] = S1
    price2[0] = S2

    k1 = np.exp(gamma1 + .5 *delta1**2) - 1
    k2 = np.exp(gamma2 + .5 *delta2**2) - 1

    for t in range(1, n+1):
        # Creating a correlated brownian motion
        dW1 = np.sqrt(dt) * np.random.normal(0, 1)
        dW3 = np.sqrt(dt) * np.random.normal(0, 1)
        dW2 = rho*dW1 + np.sqrt(1-rho**2)*dW3

        # Assuming independent jumps
        total_jumps1 = np.sum(jumps(lambda_, dt, gamma1, delta1)[0])
        total_jumps2 = np.sum(jumps(lambda_, dt, gamma1, delta1)[0])

        price1[t] = price1[t - 1] * np.exp((mu1 - .5*sig1**2 - lambda_*k1)*dt + sig1*dW1 + total_jumps1)
        price2[t] = price2[t - 1] * np.exp((mu2 - .5*sig2**2 - lambda_*k2)*dt + sig2*dW2 + total_jumps2)

    return price1, price2


if __name__ == '__main__':
    theta1 = [0.05, 0.2, 0.02, 0.005]
    theta2 = [0.05, 0.2, 0.02, 0.005]
    S1 = 100
    S2 = 100
    T = 1
    N = 1000
    lambda_ = 5

    for rho in [-1, -0.5, 0, 0.5, 1]:
        series1, series2 = MJD_2d(S1, S2, T, N, lambda_, theta1, theta2, rho)
        df = pd.DataFrame({'Asset 1': series1, 'Asset 2': series2})
        df.to_csv(f'mjd_data/mjd_data_rho={rho}.csv', index = False)

