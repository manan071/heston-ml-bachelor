import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

def nn_price(S, K, tau, kappa, theta, sigma, rho, v0, r, q, scaler_X, scaler_y, model, device):
    """Calculate the option price using the trained neural network model"""

    x  = scaler_X.transform([[np.log(K/S), tau, kappa, theta, sigma, rho, v0, r, q]])

    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        nn_price  = model(x_tensor).cpu().numpy()

    price = scaler_y.inverse_transform(nn_price)[0][0] * S

    return price

def black_scholes_call_price(S, K, tau, r, q, sigma_imp):
    """Calculate the Black-Scholes call option price"""

    d1 = (np.log(S/K) + (r-q+0.5*sigma_imp**2)*tau) / (sigma_imp*np.sqrt(tau))
    d2 = d1 - sigma_imp*np.sqrt(tau)

    call_price = S*np.exp(-q*tau)*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)

    return call_price

def implied_volatility(price, S, K, tau, r, q, lower=1e-8, upper=5.0):
    """Find implied volatility using Brent's method"""

    try:
        return brentq(lambda sigma: black_scholes_call_price(S, K, tau, r, q, sigma) - price, lower, upper)
    except ValueError:
        return np.nan

def implied_volatility_vectorized(prices, strikes, S, tau, r, q):
    """Compute implied volatility for arrays of prices and strikes"""
    implied_vols = np.array([implied_volatility(P, S, K, tau, r, q) for P, K in zip(prices, strikes)]) 

    return implied_vols

def error_metrics(model_prices, benchmark_prices, S):
    """Calculate error metrics between model prices and benchmark prices"""

    mae = np.mean(np.abs(model_prices - benchmark_prices))
    mre = np.mean(np.abs(model_prices - benchmark_prices) / np.maximum(benchmark_prices, 0.001*S))
    mse = np.mean((model_prices - benchmark_prices)**2)
    rmse = np.sqrt(mse)
    median_error = np.median(np.abs(model_prices - benchmark_prices))
    min_error = np.min(np.abs(model_prices - benchmark_prices))
    max_error = np.max(np.abs(model_prices - benchmark_prices))

    return mae, mre, mse, rmse, median_error, min_error, max_error

# Plot functions
def plot_loss(log_history):
    """Plot the training and validation loss from the log history"""

    log_history = pd.read_csv(log_history)

    plt.figure(figsize=(10, 5))
    plt.plot(log_history['Epoch'], log_history['Training Loss'], label='Training Loss')
    plt.plot(log_history['Epoch'], log_history['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.show()

def plot_volatility_smile(strikes, implied_vols, labels):
    """Plot the volatility smile for multiple curves"""

    plt.figure(figsize=(10, 5))
    for strikes, implied_vols, label in zip(strikes, implied_vols, labels):
        plt.plot(strikes, implied_vols, label=label)
    plt.xlabel('Strike $K$')
    plt.ylabel('Implied Volatility $\sigma_{imp}$')
    plt.legend(loc='best', frameon=True)
    plt.grid()
    plt.show()

def plot_prices_vs_strike(model, model_prices, benchmark_prices, strikes):
    """Plot model prices vs benchmark Heston prices"""

    plt.figure(figsize=(10, 5))
    plt.plot(strikes, model_prices, label=f'{model} Prices', marker='o')
    plt.plot(strikes, benchmark_prices, label='Benchmark Heston Prices', marker='x')
    plt.xlabel('Strike $K$')
    plt.ylabel('Option Price')
    plt.legend(loc='best', frameon=True)
    plt.grid()
    plt.show()

def plot_predicted_vs_benchmark(model, model_prices, benchmark_prices):
    """Plot predicted prices vs benchmark prices"""

    plt.figure(figsize=(10, 5))
    plt.scatter(benchmark_prices, model_prices, alpha=0.5)
    plt.plot([benchmark_prices.min(), benchmark_prices.max()], [benchmark_prices.min(), benchmark_prices.max()], 'r--')
    plt.xlabel('Benchmark Heston Prices')
    plt.ylabel(f'{model} Predicted Prices')
    plt.legend(loc='best', frameon=True)
    plt.grid()
    plt.show()


