This repository contains my source code for my bachelor thesis about Option Pricing under the Heston Model using Carr--Madan FFT and Neural Network Acceleration.

The goal of the project is to price European call options under the Heston stochastic volatility model, compare the Carr--Madan FFT method with the explicit Heston pricing formula and train neural networks to approximate prices and implied volatilities faster.

## Structure

- `heston_model.py` contains the explicit Heston pricing formula using numerical integration.
- `heston_fft.py` contains the Carr--Madan FFT implementation used to generate option prices over a grid of strikes.
- `generate_data.py` generates training data for option prices.
- `generate_data_iv.py` generates training data for implied volatilities.
- `heston_NN.py` trains the neural network for option price prediction.
- `heston_NN_iv.py` trains the neural network for implied volatility prediction.
- `NN_results.py`, `fft_results.py`, `parameter_analysis.py`, and `test_NN.py` are used to produce and compare the results shown in the thesis.
- `utils.py` contains the different utility functions used to produce results, including NN pricing, FFT interpolation, implied volatility calculation, error metrics, and plotting functions.

## Running
The dependencies are `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, and `torch`.

Typical workflow:

1. Generate data with `generate_data.py` or `generate_data_iv.py`.
2. Train a model with `heston_NN.py` or `heston_NN_iv.py`.
3. Use the result scripts to compare FFT, explicit Heston prices, neural network prices, and implied volatilities.
