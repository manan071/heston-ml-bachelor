import numpy as np

# Define the imaginary unit
i = 1j

# The characteristic function with trap and without trap
def heston_characteristic_function(phi, S, tau, kappa, theta, sigma, rho, v0, r, q, Pnum, trap): 
    x = np.log(S)
    a = kappa*theta

    if Pnum == 1:
        u = 0.5
        b = kappa - rho*sigma 
    else: 
        u = -0.5
        b = kappa
    
    d = np.sqrt((rho*sigma*i*phi-b)**2-sigma**2*(2*u*i*phi-phi**2))
    g = (b-rho*sigma*i*phi+d)/(b-rho*sigma*i*phi-d)

    if trap == 1:
        c = 1/g
        D = (b-rho*sigma*i*phi-d)/sigma**2*((1-np.exp(-d*tau))/(1-c*np.exp(-d*tau)))
        G = (1- c*np.exp(-d*tau))/(1-c)
        C = (r-q)*i*phi*tau+a/sigma**2*((b-rho*sigma*i*phi-d)*tau-2*np.log(G))
    else:
        G = (1- g*np.exp(d*tau))/(1-g)
        C = (r-q)*i*phi*tau+a/sigma**2*((b-rho*sigma*i*phi+d)*tau-2*np.log(G))
        D = (b-rho*sigma*i*phi+d)/sigma**2*((1-np.exp(d*tau))/(1-g*np.exp(d*tau)))
    
    return np.exp(C+D*v0+i*phi*x)

# The integrand for the probability calculation
def heston_integrand(phi, S, K, tau, kappa, theta, sigma, rho, v0, r, q, Pnum, trap):
    f = heston_characteristic_function(phi, S, tau, kappa, theta, sigma, rho, v0, r, q, Pnum, trap)
    
    return np.real(np.exp(-i*phi*np.log(K))*f/(i*phi))

# The probability calculation using numerical integration
def heston_probability(S, K, tau, kappa, theta, sigma, rho, v0, r, q, Pnum, trap, Lphi, Uphi, dphi):
    phi = np.arange(Lphi, Uphi, dphi)
    integrand = heston_integrand(phi, S, K, tau, kappa, theta, sigma, rho, v0, r, q, Pnum, trap)

    return 0.5+1/np.pi*np.trapezoid(integrand, phi)

# The main function to calculate the option price using the Heston model
# Use PutCall = 'C' for call option and Putcall = 'P' for put option
def heston_price(PutCall, S, K, tau, kappa, theta, sigma, rho, v0, r, q, trap, Lphi, Uphi, dphi):
    P1 = heston_probability(S, K, tau, kappa, theta, sigma, rho, v0, r, q, 1, trap, Lphi, Uphi, dphi) #The 1 indicates Pnum = 1
    P2 = heston_probability(S, K, tau, kappa, theta, sigma, rho, v0, r, q, 2, trap, Lphi, Uphi, dphi) #The 2 indicates Pnum = 2

    HestonCall = S*np.exp(-q*tau)*P1-K*np.exp(-r*tau)*P2
    HestonPut = HestonCall-S*np.exp(-q*tau)+K*np.exp(-r*tau)

    if PutCall == 'C':
        return HestonCall
    else: 
        return HestonPut





