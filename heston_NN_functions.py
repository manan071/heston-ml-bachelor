import torch
import numpy as np

def nn_price(S, K, tau, kappa, theta, sigma, rho, v0, r, q, scaler_X, scaler_y, model, device):
    """Calculate the option price using the trained neural network model."""
    
    x  = scaler_X.transform([[np.log(K/S), tau, kappa, theta, sigma, rho, v0, r, q]])

    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        nn_price  = model(x_tensor).cpu().numpy()

    price = scaler_y.inverse_transform(nn_price)[0][0] * S

    return price

