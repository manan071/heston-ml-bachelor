import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from heston_NN import HestonNN
import utils as f
import time as time

# Parameters
S=50 
tau=0.5
kappa=0.2
theta=0.05
sigma=0.3
rho=-0.7
v0=0.05
r=0.03
q=0.05

# Load the trained model and scalers
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and scalers
checkpoint = torch.load('1-heston_nn.pth', weights_only=False)
model = HestonNN(input_size=9, hidden_size=128, output_size=1).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# Plot training and validation loss
f.plot_loss('1-training_log.csv')
f.plot_loss('2-training_log.csv')
f.plot_loss('3-training_log.csv')
f.plot_loss('4-training_log.csv')
f.plot_loss('5-training_log.csv')

# Time the NN
start_time = time.time()


