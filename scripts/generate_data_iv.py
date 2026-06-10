import numpy as np

from src import heston_fft as hf
from src import utils

# Randomly save the parmeters for creatig training data
def random_parameters():
    tau = np.exp(np.random.uniform(np.log(0.01),np.log(2))) # Time to maturity

    while True:
        kappa = np.random.uniform(0.5, 5) # Mean reversion
        theta = np.random.uniform(0.01, 0.16) # Long run variance
        sigma = np.random.uniform(0.1, 0.8) # Volatility of volatility
        
        if 2 * kappa * theta > sigma**2: # Feller condition to ensure positive variance
            break

    rho = np.random.uniform(-0.9, 0.0) # Correlation
    v0 = np.random.uniform(0.01, 0.16) # Initial variance
    r = np.random.uniform(0.01, 0.05) # Risk free rate
    q = np.random.uniform(0, 0.05) # Dividend yield

    return tau, kappa, theta, sigma, rho, v0, r, q

S=1

# Loop to create training data
def generate_data(num_samples, filename):
    with open(filename, 'w', buffering=1 << 20) as f:

        for i in range(num_samples):
            tau, kappa, theta, sigma, rho, v0, r, q = random_parameters()
            strikes, prices = hf.heston_call_FFT(N=4096, eta=0.25, alpha=1.5, S=S, tau=tau, kappa=kappa, theta=theta
                        , sigma=sigma, rho=rho, v0=v0, r=r, q=q, trap=1)
            
            log_moneyness = np.log(strikes/S) # Option prices depend on relative position (moneyness) and not the absolute prices

            mask = ((log_moneyness > -0.5) & (log_moneyness < 0.5) & (prices > 0.0001) 
                    & (prices < 0.5) & np.isfinite(log_moneyness) & np.isfinite(prices)) # Filter to only include relevant data points for training
            
            strikes = strikes[mask]
            prices = prices[mask]
            log_moneyness = log_moneyness[mask]

            #Convert FFT prices to IV
            ivs = utils.implied_volatility_vectorized(prices, strikes, S, tau, r, q)

            # Filter bad IVs
            ivs_mask = np.isfinite(ivs)

            for lm, iv in zip(log_moneyness[ivs_mask], ivs[ivs_mask]):

                f.write(f"{lm},{tau},{kappa},{theta},{sigma},{rho},{v0},{r},{q},{iv}\n")

# Main block to generate data
if __name__ == "__main__":
    generate_data(20000, "heston_data_iv.txt") # Generate data for 20000 sets of parameters
