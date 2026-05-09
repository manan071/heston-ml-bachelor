import torch
import numpy as np
from heston_NN import HestonNN
from functions import nn_price

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and scalers
checkpoint = torch.load('1-heston_nn.pth', weights_only=False)
model = HestonNN(input_size=9, hidden_size=128, output_size=1).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# Test the model with specific parameters
print(nn_price(100, 110, 0.25, 2.0, 0.08, 0.3, -0.5, 0.08, 0.02, 0.01, scaler_X, scaler_y, model, device))
print(nn_price(100, 90,  1.5,  1.0, 0.12, 0.6, -0.7, 0.12, 0.04, 0.03, scaler_X, scaler_y, model, device))
print(nn_price(100, 102, 1.0,  3.0, 0.10, 0.4, -0.6, 0.10, 0.03, 0.02, scaler_X, scaler_y, model, device))
print(nn_price(100, 100, 0.75, 4.0, 0.06, 0.7, -0.9, 0.06, 0.05, 0.00, scaler_X, scaler_y, model, device))
print(nn_price(100, 95,  2.0,  0.5, 0.15, 0.2, -0.3, 0.15, 0.01, 0.04, scaler_X, scaler_y, model, device))

# Evaluate the model on the test set
"""
with torch.no_grad():
    preds   = scaler_y.inverse_transform(model(X_test).cpu().numpy()).flatten()
    actuals = scaler_y.inverse_transform(y_test.cpu().numpy()).flatten()

# Filter out very small prices
mask = actuals > 0.01  # only prices above 1% of S=1
preds_f   = preds[mask]
actuals_f = actuals[mask]

# Relative error statistics
rel_error = np.abs(preds_f - actuals_f) / actuals_f
print(f"Mean relative error:   {rel_error.mean()*100:.2f}%")
print(f"Median relative error: {np.median(rel_error)*100:.2f}%")
print(f"Max relative error:    {rel_error.max()*100:.2f}%")
print(f"Samples evaluated:     {mask.sum()}")

# Absolute error statistics
abs_error = np.abs(preds_f - actuals_f)
print(f"Mean absolute error:   {abs_error.mean()*100:.4f}")
print(f"Median absolute error: {np.median(abs_error)*100:.4f}")
print(f"Max absolute error:    {abs_error.max()*100:.4f}")
"""