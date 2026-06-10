import numpy as np

# Define the imaginary unit
i = 1j

# The characteristic function with trap and without trap
def heston_characteristic_function(u, S, tau, kappa, theta, sigma, rho, v0, r, q, Pnum, trap): 
    x = np.log(S)
    a = kappa*theta

    if Pnum == 1:
        uj = 0.5
        b = kappa - rho*sigma 
    else: 
        uj = -0.5
        b = kappa
    
    d = np.sqrt((rho*sigma*i*u-b)**2-sigma**2*(2*uj*i*u-u**2))
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

# The integrand for the probability calculation
def heston_integrand(u, S, K, tau, kappa, theta, sigma, rho, v0, r, q, Pnum, trap):
    f = heston_characteristic_function(u, S, tau, kappa, theta, sigma, rho, v0, r, q, Pnum, trap)
    
    return np.real(np.exp(-i*u*np.log(K))*f/(i*u))

# The probability calculation using numerical integration
def heston_probability(S, K, tau, kappa, theta, sigma, rho, v0, r, q, Pnum, trap, Lu, Uu, du):
    u = np.arange(Lu, Uu, du)
    integrand = heston_integrand(u, S, K, tau, kappa, theta, sigma, rho, v0, r, q, Pnum, trap)

    return 0.5+1/np.pi*np.trapezoid(integrand, u)

# The main function to calculate the option price using the Heston model
# Use PutCall = 'C' for call option and Putcall = 'P' for put option
def heston_price(PutCall, S, K, tau, kappa, theta, sigma, rho, v0, r, q, trap, Lu, Uu, du):
    P1 = heston_probability(S, K, tau, kappa, theta, sigma, rho, v0, r, q, 1, trap, Lu, Uu, du) #The 1 indicates Pnum = 1
    P2 = heston_probability(S, K, tau, kappa, theta, sigma, rho, v0, r, q, 2, trap, Lu, Uu, du) #The 2 indicates Pnum = 2

    HestonCall = S*np.exp(-q*tau)*P1-K*np.exp(-r*tau)*P2
    HestonPut = HestonCall-S*np.exp(-q*tau)+K*np.exp(-r*tau)

    if PutCall == 'C':
        return HestonCall
    else: 
        return HestonPut
