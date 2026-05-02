import numpy as np

# Define the imaginary unit
i = 1j

# The generic characteristic function for carr and madan method
def heston_characteristic_function_cm(u, S, tau, kappa, theta, sigma, rho, v0, r, q, trap): 
    x = np.log(S)
    a = kappa*theta

    u_heston = -0.5
    b = kappa
    
    d = np.sqrt((rho*sigma*i*u-b)**2-sigma**2*(2*u_heston*i*u-u**2))
    g = (b-rho*sigma*i*u+d)/(b-rho*sigma*i*u-d)

    if trap == 1:
        c = 1/g
        D = (b-rho*sigma*i*u-d)/sigma**2*((1-np.exp(-d*tau))/(1-c*np.exp(-d*tau)))
        G = (1- c*np.exp(-d*tau))/(1-c)
        C = (r-q)*i*u*tau+a/sigma**2*((b-rho*sigma*i*u-d)*tau-2*np.log(G))
    else:
        G = (1- g*np.exp(d*tau))/(1-g)
        C = (r-q)*i*u*tau+a/sigma**2*((b-rho*sigma*i*u+d)*tau-2*np.log(G))
        D = (b-rho*sigma*i*u+d)/sigma**2*((1-np.exp(d*tau))/(1-g*np.exp(d*tau)))
    
    return np.exp(C+D*v0+i*u*x)

# The integrand for the option price calculation using carr and madan method
def heston_call_integrand_cm(u, alpha, S, K, tau, kappa, theta, sigma, rho, v0, r, q, trap):
    I = np.exp(-i*u*np.log(K))*np.exp(-r*tau)*heston_characteristic_function_cm(u-(alpha+1)*i, S, tau, kappa, theta, sigma, rho, v0, r, q, trap)/(alpha**2+alpha-u**2+i*u*(2*alpha+1))
    
    return np.real(I)

# The main function to calculate the option price using the Heston model
# We use trapezoidal rule to calculate the integral. Other numerical integration methods can also be used
def heston_call_price_cm(alpha, S, K, tau, kappa, theta, sigma, rho, v0, r, q, trap, Lu, Uu, du):
    u = np.arange(Lu, Uu, du)
    integrand = heston_call_integrand_cm(u, alpha, S, K, tau, kappa, theta, sigma, rho, v0, r, q, trap)
    
    return np.exp(-alpha*np.log(K))*np.trapezoid(integrand, u)/np.pi

# Using FFT to calculate the option price for a range of strikes. The strikes and corresponding prices are returned as arrays
"""
# Trapezoidal rule for numerical integration in the FFT method
def heston_call_FFT(N, eta, alpha, S, tau, kappa, theta, sigma, rho, v0, r, q, trap):
    j = np.arange(N)
    s0 = np.log(S)
    v = np.arange(N)*eta
    lambdainc = 2*np.pi/(N*eta)
    b = N*lambdainc/2
    k = -b+j*lambdainc+s0

    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5 

    f2 = heston_characteristic_function_cm(v-(alpha+1)*i, S, tau, kappa, theta, sigma, rho, v0, r, q, trap)
    psi = np.exp(-r*tau)*f2/(alpha**2+alpha-v**2+i*v*(2*alpha+1))

    x = np.exp(i*(b-s0)*v)*psi*w
    e = np.fft.fft(x)

    prices = eta*np.exp(-alpha*k)/np.pi*np.real(e)
    strikes = np.exp(k)

    return strikes, prices
"""

# Simpson's rule for numerical integration in the FFT method
def heston_call_FFT(N, eta, alpha, S, tau, kappa, theta, sigma, rho, v0, r, q, trap):
    j = np.arange(N)
    s0 = np.log(S)
    v = np.arange(N)*eta
    lambdainc = 2*np.pi/(N*eta)
    b = N*lambdainc/2
    k = -b+j*lambdainc+s0

    # Simpson's rule weights
    w = np.ones(N)
    w[0] = 1/3
    w[-1] = 1/3
    w[1:-1:2] = 4/3  # Even j in the original indexing (odd in 0-based indexing)
    w[2:-1:2] = 2/3  # Odd j in the original indexing (even in 0-based indexing)

    f2 = heston_characteristic_function_cm(v-(alpha+1)*i, S, tau, kappa, theta, sigma, rho, v0, r, q, trap)
    psi = np.exp(-r*tau)*f2/(alpha**2+alpha-v**2+i*v*(2*alpha+1))

    x = np.exp(i*(b-s0)*v)*psi*w
    e = np.fft.fft(x)

    prices = eta*np.exp(-alpha*k)/np.pi*np.real(e)
    strikes = np.exp(k)

    return strikes, prices