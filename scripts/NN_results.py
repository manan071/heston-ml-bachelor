import torch
import numpy as np
import pandas as pd 
from sklearn.model_selection import GroupShuffleSplit
import time as time

from src.heston_NN import HestonNN
from src.heston_NN_iv import HestonNN as HestonNN_iv
from src.heston_fft import heston_call_FFT, heston_call_price_cm
from src.heston_model import heston_price
from src import utils as utils

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

# Load the trained model and scalers
# Set device
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

# Load the trained model and scalers
checkpoint = torch.load('1-heston_nn.pth', weights_only=False, map_location='cpu')
model = HestonNN(input_size=9, hidden_size=128, output_size=1).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# Plot training and validation loss
"""
utils.plot_loss(r'training_batch_size\1-training_log.csv')
utils.plot_loss(r'training_batch_size\2-training_log.csv')
utils.plot_loss(r'training_batch_size\3-training_log.csv')
utils.plot_loss(r'training_batch_size\4-training_log.csv')
utils.plot_loss(r'training_batch_size\5-training_log.csv')
"""
"""
# Plot NN prices vs FFT prices and print error metrics
parameter_sets = [

        {'name': 'Moderate benchmark case', 'tau': 0.5, 'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.7, 'v0': 0.04, 
        'r': 0.03, 'q': 0.01, 'trap': 1},

        # Short maturity with high variance
        {'name': 'Short-maturity high-variance case', 'tau': 0.25, 'kappa': 3.0, 'theta': 0.06, 'sigma': 0.4, 'rho': -0.5, 'v0': 0.06, 
        'r': 0.02, 'q': 0.02, 'trap': 1}, 

        # Long maturity with low volatility of volatility
        {'name': 'Long-maturity low-vol-of-vol case', 'tau': 1.0, 'kappa': 1.5, 'theta': 0.03, 'sigma': 0.2, 'rho': -0.3, 'v0': 0.03,
            'r': 0.04, 'q': 0.01, 'trap': 1},

        # Very short maturity with high vol-of-vol and strong negative correlation
        {'name': 'Extreme short-maturity skew case', 'tau': 0.1, 'kappa': 4.0, 'theta': 0.08, 'sigma': 0.5, 'rho': -0.8, 'v0': 0.08, 
        'r': 0.03, 'q': 0.0, 'trap': 1}, 

        # Long maturity with high variance and low mean reversion
        {'name': 'Long-maturity high-variance case', 'tau': 1.5, 'kappa': 1.0, 'theta': 0.10, 'sigma': 0.3, 'rho': -0.4, 'v0': 0.10, 
        'r': 0.02, 'q': 0.03, 'trap': 1}, 
        ]

for param_set in parameter_sets:
    strikes_fft, prices_fft = heston_call_FFT(4096, 0.25, 1.5, S, param_set['tau'], param_set['kappa'], param_set['theta'], 
                                            param_set['sigma'], param_set['rho'], param_set['v0'], param_set['r'], param_set['q'], trap=param_set['trap'])
    
    mask = (strikes_fft >= 30) & (strikes_fft <= 80) & (prices_fft > 0)
    strikes_fft = strikes_fft[mask]
    prices_fft = prices_fft[mask]

    prices_NN = np.array([utils.nn_price(S, K, param_set['tau'], param_set['kappa'], param_set['theta'], 
                                        param_set['sigma'], param_set['rho'], param_set['v0'], param_set['r'], param_set['q'], scaler_X, scaler_y, model, device) for K in strikes_fft])

    utils.plot_prices_vs_strike('NN', prices_NN, prices_fft, strikes_fft)
    utils.plot_predicted_vs_benchmark('NN', prices_NN, prices_fft)

    # Error metrics
    print(f"Error metrics for {param_set['name']}:")

    atm_band = 0.05

    # OTM options
    mask_otm = strikes_fft > S*(1+atm_band)
    mae, mre, mse, rmse, median_error, min_error, max_error = utils.error_metrics(prices_NN[mask_otm], prices_fft[mask_otm], S)

    print(f"\n-------------------------------")
    print(f"OTM options:")
    print(f"-------------------------------")
    print(f"Number of strikes: {mask_otm.sum()}")
    print("MAE:", mae)
    print("MRE:", mre)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("Median Error:", median_error)
    print("Min Error:", min_error)
    print("Max Error:", max_error)

    # Near ATM options
    mask_atm = (strikes_fft >= S*(1-atm_band)) & (strikes_fft <= S*(1+atm_band))
    mae, mre, mse, rmse, median_error, min_error, max_error = utils.error_metrics(prices_NN[mask_atm], prices_fft[mask_atm], S)

    print(f"\n-------------------------------")
    print(f"Near ATM options:")
    print(f"-------------------------------")
    print(f"Number of strikes: {mask_atm.sum()}")
    print("MAE:", mae)
    print("MRE:", mre)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("Median Error:", median_error)
    print("Min Error:", min_error)
    print("Max Error:", max_error)

    # ITM options
    mask_itm = strikes_fft < S*(1-atm_band)
    mae, mre, mse, rmse, median_error, min_error, max_error = utils.error_metrics(prices_NN[mask_itm], prices_fft[mask_itm], S)

    print(f"\n-------------------------------")
    print(f"ITM options:")
    print(f"-------------------------------")
    print(f"Number of strikes: {mask_itm.sum()}")
    print("MAE:", mae)
    print("MRE:", mre)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("Median Error:", median_error)
    print("Min Error:", min_error)
    print("Max Error:", max_error)
"""
"""
# Plot volatility smiles for the NN predictions using pricing NN
strikes_list = []
ivs_list = []
labels = []

# Rho
for rho_2 in [-0.8, -0.4, 0.0]:
    strikes, prices = heston_call_FFT(4096, 0.25, 1.5, S, tau, kappa, theta, 
                                    sigma, rho_2, v0, r, q, trap=1)

    prices_NN = np.array([utils.nn_price(S, K, tau, kappa, theta, 
                                        sigma, rho_2, v0, r, q, scaler_X, scaler_y, model, device) for K in strikes])

    mask = (strikes >= 30) & (strikes <= 80) & (prices > 0)
    strikes = strikes[mask]
    prices_NN = prices_NN[mask]

    ivs = utils.implied_volatility_vectorized(prices_NN, strikes, S, tau, r, q)
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
    
    prices_NN = np.array([utils.nn_price(S, K, tau, kappa, theta, 
                                        sigma, rho_2, v0, r, q, scaler_X, scaler_y, model, device) for K in strikes])

    mask = (strikes >= 30) & (strikes <= 80) & (prices_NN > 0)
    strikes = strikes[mask]
    prices_NN = prices_NN[mask]

    ivs = utils.implied_volatility_vectorized(prices_NN, strikes, S, tau, r, q)
    strikes_list.append(strikes)
    ivs_list.append(ivs)
    labels.append(rf'$\sigma={sigma}$')

utils.plot_volatility_smile(strikes_list, ivs_list, labels)
"""
"""
# Plot volatility smiles for the NN predictions using IV NN

# Load the trained model and scalers
checkpoint = torch.load('heston_iv_nn.pth', weights_only=False, map_location='cpu')
model = HestonNN_iv(input_size=9, hidden_size=128, output_size=1).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_X_iv = checkpoint['scaler_X']
scaler_y_iv = checkpoint['scaler_y']

strikes_list = []
ivs_list = []
labels = []

# Rho
for rho_2 in [-0.8, -0.4, 0.0]:
    strikes, prices = heston_call_FFT(4096, 0.25, 1.5, S, tau, kappa, theta, 
                                    sigma, rho_2, v0, r, q, trap=1)
    
    log_moneyness = np.log(strikes/S)

    mask = (
        (log_moneyness > np.log(30/S)) &
        (log_moneyness < np.log(80/S)) &
        np.isfinite(log_moneyness)
    )

    strikes = strikes[mask]
    log_moneyness = log_moneyness[mask]

    X = np.array([
        [lm, tau, kappa, theta, sigma, rho_2, v0, r, q]
        for lm in log_moneyness
    ])
    
    ivs_NN = np.array([model(torch.tensor(scaler_X_iv.transform([x]), 
                                        dtype=torch.float32).to(device)).detach().cpu().numpy()[0][0] for x in X])
    
    ivs_NN = scaler_y_iv.inverse_transform(ivs_NN.reshape(-1, 1)).flatten()

    strikes_list.append(strikes)
    ivs_list.append(ivs_NN)
    labels.append(rf'$\rho={rho_2}$')

utils.plot_volatility_smile(strikes_list, ivs_list, labels)

# Sigma
strikes_list.clear()
ivs_list.clear()
labels.clear()

for sigma in [0.1, 0.3, 0.6]:
    strikes, prices = heston_call_FFT(4096, 0.25, 1.5, S, tau, kappa, theta, 
                                    sigma, rho, v0, r, q, trap=1)
    
    log_moneyness = np.log(strikes/S)

    mask = (
        (log_moneyness > np.log(30/S)) &
        (log_moneyness < np.log(80/S)) &
        np.isfinite(log_moneyness)
    )

    strikes = strikes[mask]
    log_moneyness = log_moneyness[mask]

    X = np.array([
        [lm, tau, kappa, theta, sigma, rho, v0, r, q]
        for lm in log_moneyness
    ])
    
    ivs_NN = np.array([model(torch.tensor(scaler_X_iv.transform([x]), 
                                        dtype=torch.float32).to(device)).detach().cpu().numpy()[0][0] for x in X])
    
    ivs_NN = scaler_y_iv.inverse_transform(ivs_NN.reshape(-1, 1)).flatten()

    strikes_list.append(strikes)
    ivs_list.append(ivs_NN)
    labels.append(rf'$\sigma={sigma}$')

utils.plot_volatility_smile(strikes_list, ivs_list, labels)
"""
"""
# Neural network vs FFT on interpolation test cases
parameter_list = [
    # Case 1: Moderate benchmark
    (0.50, 2.00, 0.04, 0.30, -0.70, 0.04, 0.03, 0.01),

    # Case 4: Extreme short-maturity skew
    (0.10, 4.00, 0.08, 0.50, -0.80, 0.08, 0.03, 0.00),

    # Case 5: Long maturity / high variance
    (1.50, 1.00, 0.10, 0.30, -0.40, 0.10, 0.02, 0.03),
]

for tau, kappa, theta, sigma, rho, v0, r, q in parameter_list:
    strikes_fft, prices_fft = heston_call_FFT(4096, 0.25, 1.5, S, tau, kappa, theta, sigma, rho, v0, r, q, trap=1)

    mask = (strikes_fft >= 30) & (strikes_fft <= 80) & (prices_fft > 0) 
    strikes_fft = strikes_fft[mask]
    prices_fft = prices_fft[mask]

    order = np.argsort(strikes_fft)
    strikes_fft = strikes_fft[order]
    prices_fft = prices_fft[order]

    #Find strikes not in the grid 
    strikes_offgrid = (strikes_fft[:-1] + strikes_fft[1:])/2

    # Pick 20 random strikes from the off-grid stirkes
    np.random.seed(20)
    strikes_offgrid = np.random.choice(strikes_offgrid, size=20, replace=False)

    prices_fft = np.array([utils.fft_price_interpolation(S, K, tau, kappa, theta, 
                                                        sigma, rho, v0, r, q, strikes_fft, prices_fft) for K in strikes_offgrid])
    
    prices_NN = np.array([utils.nn_price(S, K, tau, kappa, theta, 
                                        sigma, rho, v0, r, q, scaler_X, scaler_y, model, device) for K in strikes_offgrid])

    explicit_prices = np.array([heston_price('C', S, K, tau, kappa, theta, sigma, rho, v0, r, q, trap=1, Lu=0.00001, Uu=50, du=0.001) for K in strikes_offgrid])

    error_fft = np.abs(prices_fft - explicit_prices)
    error_NN = np.abs(prices_NN - explicit_prices)

    print(f"Test case with tau={tau}, kappa={kappa}, theta={theta}, sigma={sigma}, rho={rho}, v0={v0}, r={r}, q={q}")
    print(f"Mean absolute error for FFT: {error_fft.mean():.4f}")
    print(f"Mean absolute error for NN: {error_NN.mean():.4f}")

    utils.plot_error_two_models('NN', 'FFT', error_NN, error_fft, strikes_offgrid)
"""
"""
# Heatmap for error across parameter space
K=50 # ATM call option as S=50

# rho vs tau
rho_values = np.linspace(-0.9, 0.0, 30)
tau_values = np.linspace(0.01, 2, 30)

error_grid = np.zeros((len(rho_values), len(tau_values)))

for i, tau_val in enumerate(tau_values):
    for j, rho_val in enumerate(rho_values):
        strikes_fft, prices_fft = heston_call_FFT(4096, 0.25, 1.5, S, tau_val, kappa, theta, 
                                                sigma, rho_val, v0, r, q, trap=1)
        
        mask = (strikes_fft >= 30) & (strikes_fft <= 80) & (prices_fft > 0)
        strikes_fft = strikes_fft[mask]
        prices_fft = prices_fft[mask]

        price_fft = np.interp(K, strikes_fft, prices_fft)

        price_NN = utils.nn_price(S, K, tau_val, kappa, theta, 
                                sigma, rho_val, v0, r, q, scaler_X, scaler_y, model, device)
        
        error_grid[i, j] = np.abs(price_NN - price_fft)

utils.plot_heatmap_error(error_grid, rho_values, tau_values, r'$\rho$', r'$\tau$')

# sigma vs tau
sigma_values = np.linspace(0.1, 0.8, 30)
tau_values = np.linspace(0.01, 2, 30)

error_grid = np.zeros((len(sigma_values), len(tau_values)))

for i, tau_val in enumerate(tau_values):
    for j, sigma_val in enumerate(sigma_values):
        strikes_fft, prices_fft = heston_call_FFT(4096, 0.25, 1.5, S, tau_val, kappa, theta, 
                                                sigma_val, rho, v0, r, q, trap=1)
        
        mask = (strikes_fft >= 30) & (strikes_fft <= 80) & (prices_fft > 0)
        strikes_fft = strikes_fft[mask]
        prices_fft = prices_fft[mask]

        price_fft = np.interp(K, strikes_fft, prices_fft)

        price_NN = utils.nn_price(S, K, tau_val, kappa, theta, 
                                sigma_val, rho, v0, r, q, scaler_X, scaler_y, model, device)
        
        error_grid[i, j] = np.abs(price_NN - price_fft)

utils.plot_heatmap_error(error_grid, sigma_values, tau_values, r'$\sigma$', r'$\tau$')
"""
"""
# Time a single option price
times = []
strikes = np.linspace(30, 80, 100)

# Carr-Madan direct integration
for K in strikes:
    for _ in range(100):
        start_time = time.perf_counter()
        price_cm = heston_call_price_cm(alpha=1.5, S=S, K=K, tau=0.5, kappa=2.0, theta=0.04, 
                                        sigma=0.3, rho=-0.7, v0=0.04, r=0.03, q=0.01, trap=1, Lu=0.00001, Uu=50, du=0.05)
        times.append(time.perf_counter()-start_time)

print(f"Mean of Carr-Madan direct integration computation time: {np.mean(times)*1000:.4f} ms") 
print(f"Standard deviation of Carr-Madan direct integration computation time: {np.std(times)*1000:.4f} ms")

times.clear()
# FFT
for K in strikes:
    for _ in range(100):
        start_time = time.perf_counter()
        strikes_fft, prices_fft = heston_call_FFT(4096, 0.25, 1.5, S, tau=0.5, kappa=2.0, theta=0.04, 
                                    sigma=0.3, rho=-0.7, v0=0.04, r=0.03, q=0.01, trap=1)
        price_fft = np.interp(K, strikes_fft, prices_fft)
        times.append(time.perf_counter()-start_time)

print(f"Mean of FFT computation time: {np.mean(times)*1000:.4f} ms") 
print(f"Standard deviation of FFT computation time: {np.std(times)*1000:.4f} ms")

times.clear()
# NN
_ = utils.nn_price_batch(S, strikes, 0.5, 2.0, 0.04, 0.3, -0.7, 0.04, 0.03, 0.01, scaler_X, scaler_y, model, device) # Warmup 

for K in strikes:
    for _ in range(100):
        start_time = time.perf_counter()
        price_NN = utils.nn_price(S, K, 0.5, 2.0, 0.04, 
                                    0.3, -0.7, 0.04, 0.03, 0.01, scaler_X, scaler_y, model, device)
        times.append(time.perf_counter()-start_time)

print(f"Mean of NN computation time: {np.mean(times)*1000:.4f} ms") 
print(f"Standard deviation of NN computation time: {np.std(times)*1000:.4f} ms")
"""
"""
# Time the FFT and NN for 10 maturity with 20 strikes each
maturities = np.linspace(0.1, 2.0, 10)
strikes_grid = np.linspace(30, 80, 20)

fft_prices_grid = np.zeros((len(maturities), len(strikes_grid)))
nn_prices_grid = np.zeros((len(maturities), len(strikes_grid)))
iv_surface_iv_nn = np.zeros((len(maturities), len(strikes_grid)))

# FFT
fft_times = []
for _ in range(100):
    start_time = time.perf_counter()
    for i, tau_i in enumerate(maturities):
        strikes_fft, prices_fft = heston_call_FFT(4096, 0.25, 1.5, S, tau_i, kappa, theta,
                                                sigma, rho, v0, r, q, trap=1)
        fft_prices_grid[i, :] = np.interp(strikes_grid, strikes_fft, prices_fft)
    fft_times.append(time.perf_counter() - start_time)

print(f"Mean of FFT computation time (10 maturities x 20 strikes): {np.mean(fft_times)*1000:.4f} ms")
print(f"Standard deviation of FFT computation time: {np.std(fft_times)*1000:.4f} ms")

# NN
_ = utils.nn_price_batch(S, strikes_grid, maturities[0], kappa, theta, sigma, rho, v0, r, q,
                        scaler_X, scaler_y, model, device)

nn_times = []
for _ in range(100):
    start_time = time.perf_counter()
    for i, tau_i in enumerate(maturities):
        nn_prices_grid[i, :] = utils.nn_price_batch(S, strikes_grid, tau_i, kappa, theta, sigma, rho, v0, r, q,
                                                    scaler_X, scaler_y, model, device)
    nn_times.append(time.perf_counter() - start_time)

print(f"Mean of NN computation time (10 maturities x 20 strikes): {np.mean(nn_times)*1000:.4f} ms")
print(f"Standard deviation of NN computation time: {np.std(nn_times)*1000:.4f} ms")

# IV NN
checkpoint_iv = torch.load('heston_iv_nn.pth', weights_only=False, map_location='cpu')
model_iv = HestonNN_iv(input_size=9, hidden_size=128, output_size=1).to(device)
model_iv.load_state_dict(checkpoint_iv['model_state_dict'])
model_iv.eval()

scaler_X_iv = checkpoint_iv['scaler_X']
scaler_y_iv = checkpoint_iv['scaler_y']

_ = utils.nn_iv_batch(S, strikes_grid, maturities[0], kappa, theta, sigma, rho, v0, r, q,
                    scaler_X_iv, scaler_y_iv, model_iv, device)

iv_nn_times = []
for _ in range(100):
    start_time = time.perf_counter()
    for i, tau_i in enumerate(maturities):
        iv_surface_iv_nn[i, :] = utils.nn_iv_batch(S, strikes_grid, tau_i, kappa, theta, sigma, rho, v0, r, q,
                                                    scaler_X_iv, scaler_y_iv, model_iv, device)
    iv_nn_times.append(time.perf_counter() - start_time)

print(f"Mean of IV NN computation time (10 maturities x 20 strikes): {np.mean(iv_nn_times)*1000:.4f} ms")
print(f"Standard deviation of IV NN computation time: {np.std(iv_nn_times)*1000:.4f} ms")

# Compute implied volatility surface
iv_surface_fft = np.zeros((len(maturities), len(strikes_grid)))
iv_surface_nn = np.zeros((len(maturities), len(strikes_grid)))

for i, tau_i in enumerate(maturities):
    iv_surface_fft[i, :] = utils.implied_volatility_vectorized(fft_prices_grid[i, :], strikes_grid, S, tau_i, r, q)
    iv_surface_nn[i, :] = utils.implied_volatility_vectorized(nn_prices_grid[i, :], strikes_grid, S, tau_i, r, q)

utils.plot_volatility_surface(strikes_grid, maturities, iv_surface_fft, 'FFT')
utils.plot_volatility_surface(strikes_grid, maturities, iv_surface_nn, 'Pricing NN')
utils.plot_volatility_surface(strikes_grid, maturities, iv_surface_iv_nn, 'IV NN')
"""
"""
# Time full surface batch with pre batching 
maturities = np.linspace(0.1, 2.0, 10)
strikes_grid = np.linspace(30, 80, 20)

# FFT
fft_prices_grid = np.zeros((len(maturities), len(strikes_grid)))

fft_times = []
for _ in range(100):
    start_time = time.perf_counter()
    for i, tau_i in enumerate(maturities):
        strikes_fft, prices_fft = heston_call_FFT(4096, 0.25, 1.5, S, tau_i, kappa, theta,
                                                sigma, rho, v0, r, q, trap=1)
        fft_prices_grid[i, :] = np.interp(strikes_grid, strikes_fft, prices_fft)
    fft_times.append(time.perf_counter() - start_time)

# NN

# Build input for warmup
all_strikes = np.tile(strikes_grid, len(maturities))
all_taus = np.repeat(maturities, len(strikes_grid))
n = len(all_strikes)
X = np.empty((n, 9))
X[:, 0] = np.log(all_strikes / S)
X[:, 1] = all_taus
X[:, 2] = 2.0; X[:, 3] = 0.04; X[:, 4] = 0.3
X[:, 5] = -0.7; X[:, 6] = 0.04; X[:, 7] = 0.03; X[:, 8] = 0.01
X = ((X - scaler_X.mean_) / scaler_X.scale_).astype(np.float32)

_ = model(torch.from_numpy(X).to(device))  # warmup

nn_times = []
for _ in range(100):
    start_time = time.perf_counter()
    all_strikes = np.tile(strikes_grid, len(maturities))
    all_taus = np.repeat(maturities, len(strikes_grid))

    n = len(all_strikes)
    X = np.empty((n, 9))
    X[:, 0] = np.log(all_strikes / S)
    X[:, 1] = all_taus
    X[:, 2] = 2.0; X[:, 3] = 0.04; X[:, 4] = 0.3
    X[:, 5] = -0.7; X[:, 6] = 0.04; X[:, 7] = 0.03; X[:, 8] = 0.01
    X = ((X - scaler_X.mean_) / scaler_X.scale_).astype(np.float32)

    with torch.no_grad():
        prices = model(torch.from_numpy(X).to(device)).cpu().numpy()
    nn_times.append(time.perf_counter() - start_time)

print(f"Mean of FFT computation time (10 maturities x 20 strikes): {np.mean(fft_times)*1000:.4f} ms")
print(f"Standard deviation of FFT computation time: {np.std(fft_times)*1000:.4f} ms")

print(f"Mean of NN computation time (10 maturities x 20 strikes): {np.mean(nn_times)*1000:.4f} ms")
print(f"Standard deviation of NN computation time: {np.std(nn_times)*1000:.4f} ms")
"""
"""
# Error across moneyness using test dataset 
atm_band=0.05

# Load the generated data
data = np.loadtxt('heston_data.txt', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

# Use the same grouped split as heston_NN.py to prevent data leakage
group_cols = [1, 2, 3, 4, 5, 6, 7, 8]
df_X = pd.DataFrame(X)
groups = df_X.groupby(group_cols).ngroup().values

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

X_temp, y_temp = X[temp_idx], y[temp_idx]
groups_temp = groups[temp_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))

X_test = X_temp[test_idx]
y_test = y_temp[test_idx]

# Predict on the full held-out test set
X_test_scaled = scaler_X.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    preds_scaled = model(X_test_tensor).cpu().numpy()

preds = scaler_y.inverse_transform(preds_scaled).flatten()
actuals = y_test

# Convert normalized prices from S=1 to the thesis scale S=50
preds_scaled_to_S = preds * S
actuals_scaled_to_S = actuals * S

log_moneyness = X_test[:, 0]
lower_atm = np.log(1 - atm_band)
upper_atm = np.log(1 + atm_band)

regions = {
    'OTM': log_moneyness > upper_atm,
    'Near ATM': (log_moneyness >= lower_atm) & (log_moneyness <= upper_atm),
    'ITM': log_moneyness < lower_atm,
}

rows = []

for region_name, mask in regions.items():
    region_preds = preds_scaled_to_S[mask]
    region_actuals = actuals_scaled_to_S[mask]

    abs_error = np.abs(region_preds - region_actuals)
    rel_error = abs_error / region_actuals
    rmse = np.sqrt(np.mean((region_preds - region_actuals) ** 2))

    rows.append({
        'Region': region_name,
        'Samples': mask.sum(),
        'MAE': abs_error.mean(),
        'Median AE': np.median(abs_error),
        'RMSE': rmse,
        'MRE (%)': rel_error.mean() * 100,
        '95% AE': np.percentile(abs_error, 95),
        '99% AE': np.percentile(abs_error, 99),
        'Max AE': abs_error.max(),
    })

results = pd.DataFrame(rows)

print('\nHeld-out test set errors grouped by moneyness')
print(f'Absolute errors are rescaled to S={S}')
print(f'Near ATM is defined as {1-atm_band:.2f} <= K/S <= {1+atm_band:.2f}\n')
print(results.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
"""
