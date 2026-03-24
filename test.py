import numpy as np
import heston_model as hm
import heston_fft as heston_fft

# Test of the Heston model implementation with given parameters
price_testP = hm.heston_price(PutCall='P', S=100, K=100, tau=0.5, kappa=5, theta=0.05, sigma=0.5, rho=-0.8, v0=0.05, 
                r=0.03, q=0.02, trap=1, Lphi=0.00001, Uphi=50, dphi=0.001)
print(price_testP)

price_testC = hm.heston_price(PutCall='C', S=100, K=100, tau=0.5, kappa=5, theta=0.05, sigma=0.5, rho=-0.8, v0=0.05, 
                r=0.03, q=0.02, trap=1, Lphi=0.00001, Uphi=50, dphi=0.001)
print(price_testC)

# Test of Heston model implementation for car and madan method with given parameters
S = 100*np.exp(-0.02*0.5) # We adjust the spot price for dividend yield (not included in the carr and method method)

price_testC_cm = heston_fft.heston_call_price_cm(alpha=1.5, S=S, K=100, tau=0.5, kappa=5, theta=0.05, sigma=0.5, rho=-0.8, v0=0.05, 
                r=0.03, trap=1, Lu=0.00001, Uu=50, du=0.001)
print(price_testC_cm)

# Test of FFT
callFFT = heston_fft.heston_call_FFT(N=1024, eta=0.0977, alpha=1.5, S=50, tau=0.5, kappa=0.2, theta=0.05, sigma=0.3, rho=-0.7, v0=0.05, r=0.03, trap=1)
print(callFFT[0:10])