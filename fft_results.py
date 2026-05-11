import numpy as np
import matplotlib.pyplot as plt
from heston_fft import heston_call_FFT
from heston_model import heston_price
import utils as utils 
import time as time

# Parameters
S=50 
tau=0.5
kappa=0.2
theta=0.05
sigma=0.3
rho=-0.7
v0=0.05
r=0.03
q=0.05

# Plot FFT prices vs Heston explicit prices
strikes_fft, prices_fft = heston_call_FFT(4096, 0.25, 1.5, S, tau, kappa, theta, 
                                          sigma, rho, v0, r, q, trap=1) 

# Filter 
mask = (strikes_fft >= 30) & (strikes_fft <= 130) & (prices_fft > 0)
strikes_fft = strikes_fft[mask]
prices_fft = prices_fft[mask]

prices_explicit = np.array([heston_price('C', S, K, tau, kappa, theta, 
                                         sigma, rho, v0, r, q, trap=1, Lu=0.00001, Uu=50, du=0.001) for K in strikes_fft])

utils.plot_prices_vs_strike('FFT', prices_fft, prices_explicit, strikes_fft)
utils.plot_predicted_vs_benchmark('FFT', prices_fft, prices_explicit)

# Plot implied volatilities
strikes_list = []
ivs_list = []
labels = []

# Rho
for rho_2 in [-0.8, -0.4, 0.0]:
    strikes, prices = heston_call_FFT(4096, 0.25, 1.5, S, tau, kappa, theta, 
                                      sigma, rho_2, v0, r, q, trap=1)

    mask = (strikes >= 30) & (strikes <= 130) & (prices > 0)
    strikes = strikes[mask]
    prices = prices[mask]

    ivs = utils.implied_volatility_vectorized(prices, strikes, S, tau, r, q)
    strikes_list.append(strikes)
    ivs_list.append(ivs)
    labels.append(rf'$\rho={rho_2}$')

utils.plot_volatility_smile(strikes_list, ivs_list, labels)

# Sigma
strikes_list.clear()
ivs_list.clear()
labels.clear()

for sigma in [0.1, 0.3, 0.6]:
    strikes, prices = heston_call_FFT(4096, 0.25, 1.5, S, tau, kappa, theta, 
                                      sigma, rho, v0, r, q, trap=1)
    
    mask = (strikes >= 30) & (strikes <= 130) & (prices > 0)
    strikes = strikes[mask]
    prices = prices[mask]

    ivs = utils.implied_volatility_vectorized(prices, strikes, S, tau, r, q)
    strikes_list.append(strikes)
    ivs_list.append(ivs)
    labels.append(rf'$\sigma={sigma}$')

utils.plot_volatility_smile(strikes_list, ivs_list, labels)

# Time the FFT 
parameter_list = [
    (0.5, 2.0, 0.04, 0.3, -0.7, 0.04, 0.03, 0.01), 
    (0.1, 5.0, 0.01, 0.8, -0.9, 0.01, 0.05, 0.0),    # Short maturity
    (2.0, 0.5, 0.16, 0.1, -0.1, 0.16, 0.01, 0.05),   # Long maturity
]

times = []

for tau, kappa, theta, sigma, rho, v0, r, q in parameter_list:
    for _ in range(100):
        start_time = time.perf_counter()
        strikes, prices = heston_call_FFT(4096, 0.25, 1.5, S, tau, kappa, theta, 
                                          sigma, rho, v0, r, q, trap=1)
        times.append(time.perf_counter() - start_time)

print(f"Mean of FFT computation time: {np.mean(times)*1000:.4f} ms") 
print(f"Standard deviation of FFT computation time: {np.std(times)*1000:.4f} ms")