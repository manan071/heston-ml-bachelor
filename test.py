import numpy as np
import heston_model as hm
import heston_fft as hfft
import generate_data as gd

# Test of the Heston model implementation with given parameters
price_testP = hm.heston_price(PutCall='P', S=100, K=100, tau=0.5, kappa=5, theta=0.05, sigma=0.5, rho=-0.8, v0=0.05, 
                r=0.03, q=0.02, trap=1, Lu=0.00001, Uu=50, du=0.001)
print(price_testP)

price_testC = hm.heston_price(PutCall='C', S=100, K=100, tau=0.5, kappa=5, theta=0.05, sigma=0.5, rho=-0.8, v0=0.05, 
                r=0.03, q=0.02, trap=1, Lu=0.00001, Uu=50, du=0.001)
print(price_testC)

print()

# Test of Heston model implementation for car and madan method with given parameters
price_testC_cm = hfft.heston_call_price_cm(alpha=1.5, S=100, K=100, tau=0.5, kappa=5, theta=0.05, sigma=0.5, rho=-0.8, v0=0.05, 
                r=0.03, q=0.02, trap=1, Lu=0.00001, Uu=50, du=0.001)
print(f"Call option price (Car and Madan): {price_testC_cm}")

print()

# Test of FFT implementation of Heston model for call option with given parameters against Rouah
callFFT = hfft.heston_call_FFT(N=1024, eta=0.0977, alpha=1.5, S=50, tau=0.5, kappa=0.2, theta=0.05, 
                             sigma=0.3, rho=-0.7, v0=0.05, r=0.03, q=0.05, trap=1)
print("Strikes and corresponding call option prices using FFT:")
print(callFFT[0][509:516], callFFT[1][509:516])

# ---------------------------------------------------------------------------
"""
# Test for the data generation process for the NN
tau, kappa, theta, sigma, rho, v0, r, q = gd.random_parameters()
S=1

strikes, prices = hfft.heston_call_FFT(N=1024, eta=0.0977, alpha=1.5, S=S, tau=tau, kappa=kappa, theta=theta
                   , sigma=sigma, rho=rho, v0=v0, r=r, q=q, trap=1)

log_moneyness = np.log(strikes/S)

print(log_moneyness[509:516], prices[509:516])

print()

# Test for the data generation process
print(gd.data.shape)
print(gd.data.min(axis=0))
print(gd.data.max(axis=0))
print(gd.data.mean(axis=0)) 

print()

# Test Call option price for NN with 5 set of parameters

print(hm.heston_price(PutCall='C', S=100, K=110, tau=0.25, kappa=2.0, theta=0.08, sigma=0.3, rho=-0.5, v0=0.08, 
                      r=0.02, q=0.01, trap=1, Lu=0.00001, Uu=50, du=0.001))

print(hm.heston_price(PutCall='C', S=100, K=90, tau=1.5, kappa=1.0, theta=0.12, sigma=0.6, rho=-0.7, v0=0.12, 
                      r=0.04, q=0.03, trap=1, Lu=0.00001, Uu=50, du=0.001))

print(hm.heston_price(PutCall='C', S=100, K=102, tau=1.0, kappa=3.0, theta=0.10, sigma=0.4, rho=-0.6, v0=0.10, 
                      r=0.03, q=0.02, trap=1, Lu=0.00001, Uu=50, du=0.001))

print(hm.heston_price(PutCall='C', S=100, K=100, tau=0.75, kappa=4.0, theta=0.06, sigma=0.7, rho=-0.9, v0=0.06, 
                      r=0.05, q=0.00, trap=1, Lu=0.00001, Uu=50, du=0.001))

print(hm.heston_price(PutCall='C', S=100, K=95, tau=2.0, kappa=0.5, theta=0.15, sigma=0.2, rho=-0.3, v0=0.15, 
                      r=0.01, q=0.04, trap=1, Lu=0.00001, Uu=50, du=0.001))
"""
