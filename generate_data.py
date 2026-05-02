import numpy as np
import heston_fft as hf

# Randomly save the parmeters for creatig training data
def random_parameters():
    tau = np.exp(np.random.uniform(np.log(0.01),np.log(2))) # Time to maturity
    kappa = np.random.uniform(0.5, 5) # Mean reversion
    theta= np.random.uniform(0.01, 0.16) # Long run variance
    sigma = np.random.uniform(0.1, 0.8) # Volatility of volatility
    rho = np.random.uniform(-1, -0.1) # Correlation
    v0 = np.random.uniform(0.01, 0.16) # Initial variance
    r = np.random.uniform(0.01, 0.05) # Risk free rate
    q = np.random.uniform(0, 0.05) # Dividend yield

    return tau, kappa, theta, sigma, rho, v0, r, q

S=1

rows = []

# Loop to create training data
for i in range(20000):
    tau, kappa, theta, sigma, rho, v0, r, q = random_parameters()
    strikes, prices = hf.heston_call_FFT(N=4096, eta=0.25, alpha=1.5, S=S, tau=tau, kappa=kappa, theta=theta
                   , sigma=sigma, rho=rho, v0=v0, r=r, q=q, trap=1)
    
    log_moneyness = np.log(strikes/S) # Option prices depend on relative position (moneyness) and not the absolute prices

    filter = ((log_moneyness > -0.5) & (log_moneyness < 0.5) & (prices > 0.0001) 
              & (prices < 0.5) & np.isfinite(log_moneyness) & np.isfinite(prices)) # Filter to only include relevant data points for training

    for lm, p in zip(log_moneyness[filter], prices[filter]):
        rows.append([lm, tau, kappa, theta, sigma, rho, v0, r, q, p])

data = np.array(rows)

np.savetxt("heston_data.txt", data, delimiter=",")