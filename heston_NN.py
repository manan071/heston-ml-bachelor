import numpy as np
import generate_data as gd
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Load the generated data
X = gd.data[:,:-1]
y = gd.data[:,-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1) # Random state is set for reproducibility

# ---------------------------------------------------------------------------
# MLP for predication 
class HestonNN(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)
    
# Create the model
model = HestonNN()

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) # Reshape y to be a column vector

# Train the model
for epoch in range(110):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = loss_function(predictions, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Test the model 
model.eval()

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    test_predictions = model(X_test) 
    test_loss = loss_function(test_predictions, y_test)

print(f"Test loss: {test_loss.item()}")

# Test the model with specific parameters
test_input = torch.tensor([[np.log(100/100), 0.5, 5.0, 0.05, 0.5, -0.8, 0.05, 0.03, 0.02]], dtype=torch.float32)

with torch.no_grad():
    nn_price  = model(test_input)

print(nn_price.item()*100)

