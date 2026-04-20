import torch
import numpy as np
from heston_NN import HestonNN

checkpoint = torch.load('heston_nn.pth', weights_only=False)
model = HestonNN(hidden_size=256)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

def nn_price(S, K, tau, kappa, theta, sigma, rho, v0, r, q):
    x  = scaler_X.transform([[np.log(K/S), tau, kappa, theta, sigma, rho, v0, r, q]])
    nn = scaler_y.inverse_transform(model(torch.tensor(x, dtype=torch.float32)).detach().numpy())[0][0] * S
    print(f"NN price: {nn:.4f}")

nn_price(100, 110, 0.25, 2.0, 0.08, 0.3, -0.5, 0.08, 0.02, 0.01)
nn_price(100, 90,  1.5,  1.0, 0.12, 0.6, -0.7, 0.12, 0.04, 0.03)
nn_price(100, 102, 1.0,  3.0, 0.10, 0.4, -0.6, 0.10, 0.03, 0.02)
nn_price(100, 100, 0.75, 4.0, 0.06, 0.7, -0.9, 0.06, 0.05, 0.00)
nn_price(100, 95,  2.0,  0.5, 0.15, 0.2, -0.3, 0.15, 0.01, 0.04)