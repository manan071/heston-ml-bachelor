import numpy as np
import matplotlib.pyplot as plt
import time

from src import heston_fft as hfft
from src import heston_model as hm

# Baseline parameter test cases
testcases = [

        # ATM option
        {'name': 'ATM option', 'S': 100, 'K': 100, 'tau': 0.5, 'kappa': 5, 'theta': 0.05, 'sigma': 0.5, 'rho': -0.8, 'v0': 0.05, 
         'r': 0.03, 'q': 0.02, 'trap': 1},

        # ITM option
        {'name': 'ITM option', 'S': 100, 'K': 80, 'tau': 0.5, 'kappa': 5, 'theta': 0.05, 'sigma': 0.5, 'rho': -0.8, 'v0': 0.05, 
         'r': 0.03, 'q': 0.02, 'trap': 1}, 

        # OTM option
        {'name': 'OTM option', 'S': 100, 'K': 120, 'tau': 0.5, 'kappa': 5, 'theta': 0.05, 'sigma': 0.5, 'rho': -0.8, 'v0': 0.05,
            'r': 0.03, 'q': 0.02, 'trap': 1},

        # Long maturity option
        {'name': 'Long maturity option', 'S': 100, 'K': 100, 'tau': 2.0, 'kappa': 5, 'theta': 0.05, 'sigma': 0.5, 'rho': -0.8, 'v0': 0.05, 
         'r': 0.03, 'q': 0.02, 'trap': 1}, 

        # High vol of vol
        {'name': 'High vol of vol', 'S': 100, 'K': 100, 'tau': 2.0, 'kappa': 5, 'theta': 0.05, 'sigma': 1, 'rho': -0.8, 'v0': 0.05, 
         'r': 0.03, 'q': 0.02, 'trap': 1}, 
         ]

# 1. Analysis of alpha
alphas = [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
alpha_errors = []

# 2. Analysis of N
N_values = [256, 512, 1024, 2048, 4096, 8192]
N_errors = []
N_times = []

# 3. Analysis of eta
etas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
eta_errors = []

# Benchmark price from Gil-Pelaez
for case in testcases:
    S = case['S']
    K = case['K']
    tau = case['tau']
    kappa = case['kappa']
    theta = case['theta']
    sigma = case['sigma']
    rho = case['rho']
    v0 = case['v0']
    r = case['r']
    q = case['q']
    trap = case['trap']

    benchmark_price = hm.heston_price('C', S, K, tau, kappa, theta, sigma, rho, v0, r, q, trap, 
                           Lu=0.00001, Uu=50, du=0.001)
    
    print(f"\nTest case: {case['name']}")
    print(f"Benchmark (Gil-Pelaez) price: {benchmark_price:.6f}")

    # 1. Analysis of alpha
    case_alpha_errors = []

    for alpha in alphas:
        strikes, prices = hfft.heston_call_FFT(4096, 0.25, alpha, S, tau, kappa, theta, sigma, rho, v0, r, q, trap)
        fft_price = np.interp(K, strikes, prices)
        case_alpha_errors.append(abs(benchmark_price - fft_price))

    alpha_errors.append(case_alpha_errors)

    # 2. Analysis of N
    case_N_errors = []
    case_N_times = []

    for N in N_values:
        start = time.time()
        strikes, prices = hfft.heston_call_FFT(N, 0.25, 1.5, S, tau, kappa, theta, sigma, rho, v0, r, q, trap)
        elapsed = time.time() - start
        fft_price = np.interp(K, strikes, prices)
        case_N_errors.append(abs(benchmark_price - fft_price))
        case_N_times.append(elapsed)

    N_errors.append(case_N_errors)
    N_times.append(case_N_times)

    # 3. Analysis of eta
    case_eta_errors = []

    for eta in etas:
        strikes, prices = hfft.heston_call_FFT(4096, eta, 1.5, S, tau, kappa, theta, sigma, rho, v0, r, q, trap)
        fft_price = np.interp(K, strikes, prices)
        case_eta_errors.append(abs(benchmark_price - fft_price))

    eta_errors.append(case_eta_errors)


# Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Alpha plot
for i, case in enumerate(testcases):
    axes[0].plot(alphas, alpha_errors[i], 'o-', label=case['name'])

axes[0].set_xlabel(r'$\alpha$')
axes[0].set_ylabel('Absolute Error')
axes[0].set_title(r'Error vs $\alpha$', fontsize=12)
axes[0].set_yscale('log')
axes[0].legend(fontsize=9, loc='best', frameon=True)

# N plot
for i, case in enumerate(testcases):
    axes[1].plot(N_values, N_errors[i], 'o-', label=case['name'])

axes[1].set_xlabel('$N$')
axes[1].set_ylabel('Absolute Error')
axes[1].set_title('Error vs $N$', fontsize=12)
axes[1].set_yscale('log')
axes[1].legend(fontsize=9, loc='best', frameon=True)

# Eta plot
for i, case in enumerate(testcases):
    axes[2].plot(etas, eta_errors[i], 'o-', label=case['name'])

axes[2].set_xlabel(r'$\eta$')
axes[2].set_ylabel('Absolute Error')
axes[2].set_title(r'Error vs $\eta$', fontsize=12)
axes[2].set_yscale('log')
axes[2].legend(fontsize=9, loc='best', frameon=True)

plt.tight_layout()
plt.show()


# Print tables
for i, case in enumerate(testcases):

    print(f"\n-------------------------------")
    print(f"Test case: {case['name']}")
    print(f"-------------------------------")

    print("\n--- Alpha Analysis ---")
    print(f"{'Alpha':<10} {'Error':<15}")
    for a, e in zip(alphas, alpha_errors[i]):
        print(f"{a:<10.2f} {e:<15.2e}")

    print("\n--- N Analysis ---")
    print(f"{'N':<10} {'Error':<15} {'Time (s)':<10}")
    for n, e, t in zip(N_values, N_errors[i], N_times[i]):
        print(f"{n:<10} {e:<15.2e} {t:<10.4f}")

    print("\n--- Eta Analysis ---")
    print(f"{'Eta':<10} {'Error':<15}") 
    for eta, e in zip(etas, eta_errors[i]):
        print(f"{eta:<10.2f} {e:<15.2e}")
