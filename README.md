This repository contains my source code for my bachelor thesis about Option Pricing under the Heston Model using Carr--Madan FFT and Neural Network Acceleration.

The goal of the project is to price European call options under the Heston stochastic volatility model, compare the Carr--Madan FFT method with the explicit Heston pricing formula and train neural networks to approximate prices and implied volatilities faster.

## Structure

- `src/heston_model.py` contains the explicit Heston pricing formula using numerical integration.
- `src/heston_fft.py` contains the Carr--Madan FFT implementation used to generate option prices over a grid of strikes.
- `src/heston_NN.py` trains the neural network for option price prediction.
- `src/heston_NN_iv.py` trains the neural network for implied volatility prediction.
- `src/utils.py` contains the different utility functions used to produce results, including NN pricing, FFT interpolation, implied volatility calculation, error metrics, and plotting functions.
- `scripts/generate_data.py` generates training data for option prices.
- `scripts/generate_data_iv.py` generates training data for implied volatilities.
- `scripts/NN_results.py`, `scripts/fft_results.py`, `scripts/parameter_analysis.py`, and `scripts/test_NN.py` are used to produce and compare the results shown in the thesis.

## Running
The dependencies are `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, and `torch`.
Run the files as modules from the project root so the imports and data paths work correctly.

Typical workflow:

1. Generate data with `python -m scripts.generate_data` or `python -m scripts.generate_data_iv`.
2. Train a model with `python -m src.heston_NN` or `python -m src.heston_NN_iv`.
3. Use the result scripts to compare FFT, explicit Heston prices, neural network prices, and implied volatilities.
