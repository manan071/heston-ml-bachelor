import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from heston_NN import HestonNN
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

# Load the trained model and scalers
# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

# Load the trained model and scalers
checkpoint = torch.load('1-heston_nn.pth', weights_only=False)
model = HestonNN(input_size=9, hidden_size=128, output_size=1).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# Plot training and validation loss
"""
utils.plot_loss('1-training_log.csv')
utils.plot_loss('2-training_log.csv')
utils.plot_loss('3-training_log.csv')
utils.plot_loss('4-training_log.csv')
utils.plot_loss('5-training_log.csv')
"""

# Plot NN prices vs FFT prices
strikes_fft, prices_fft = heston_call_FFT(4096, 0.25, 1.5, S, tau, kappa, theta, 
                                          sigma, rho, v0, r, q, trap=1) 

# Filter 
mask = (strikes_fft >= 30) & (strikes_fft <= 130) & (prices_fft > 0)
strikes_fft = strikes_fft[mask]
prices_fft = prices_fft[mask]

prices_NN = np.array([model(torch.tensor(scaler_X.transform([[np.log(K/S), tau, kappa, theta, sigma, rho, v0, r, q]]), 
                                         dtype=torch.float32).to(device)).detach().cpu().numpy()[0][0] for K in strikes_fft])

prices_NN = scaler_y.inverse_transform(prices_NN.reshape(-1, 1)).flatten()*S

utils.plot_prices_vs_strike('NN', prices_NN, prices_fft, strikes_fft)

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
    
    mask = (strikes_fft >= 30) & (strikes_fft <= 130) & (prices_fft > 0)
    strikes_fft = strikes_fft[mask]
    prices_fft = prices_fft[mask]

    prices_NN = np.array([model(torch.tensor(scaler_X.transform([[np.log(K/S), param_set['tau'], param_set['kappa'], param_set['theta'], param_set['sigma'], param_set['rho'], param_set['v0'], param_set['r'], param_set['q']]]), 
                                         dtype=torch.float32).to(device)).detach().cpu().numpy()[0][0] for K in strikes_fft])

    prices_NN = scaler_y.inverse_transform(prices_NN.reshape(-1, 1)).flatten()*S

    utils.plot_prices_vs_strike('NN', prices_NN, prices_fft, strikes_fft)
    utils.plot_predicted_vs_benchmark('NN', prices_NN, prices_fft)


# Error metrics
atm_band = 0.01
# OTM options
mask_otm = strikes_fft > S*(1+atm_band)
mae, mre, mse, rmse, median_error, min_error, max_error = utils.error_metrics(prices_NN[mask_otm], prices_fft[mask_otm], S)

print(f"\n-------------------------------")
print(f"OTM options:")
print(f"-------------------------------")
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
print("MAE:", mae)
print("MRE:", mre)
print("MSE:", mse)
print("RMSE:", rmse)
print("Median Error:", median_error)
print("Min Error:", min_error)
print("Max Error:", max_error)
