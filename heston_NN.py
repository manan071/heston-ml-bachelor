import numpy as np
import generate_data as gd
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# MLP for prediction 
class HestonNN(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Random seed for reproducibility
    np.random.seed(50)
    torch.manual_seed(50)

    # Load the generated data
    X = gd.data[:,:-1]
    y = gd.data[:,-1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1) 

    # Standardize the features and target variable
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1,1))

    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1,1))

    # Create the model
    model = HestonNN()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) # Reshape y to be a column vector

    # Train the model
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
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
        test_loss = criterion(test_predictions, y_test)

    print(f"Test loss: {test_loss.item()}")

    # Save the model and scalers
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, 'heston_nn.pth')
    print("Model saved.")

    print()

    # Test the model with specific parameters
    test_input_standardized = scaler_X.transform([[np.log(100/100), 0.5, 5.0, 0.05, 0.5, -0.8, 0.05, 0.03, 0.02]])
    test_input = torch.tensor(test_input_standardized, dtype=torch.float32)

    with torch.no_grad():
        nn_price  = model(test_input)
        nn_price = scaler_y.inverse_transform(nn_price.numpy())

    print(nn_price[0][0]*100)

    print()

    with torch.no_grad():
        preds   = scaler_y.inverse_transform(model(X_test).numpy()).flatten()
        actuals = scaler_y.inverse_transform(y_test.numpy()).flatten()

    # Filter out very small prices
    mask = actuals > 0.01  # only prices above 1% of S=1
    preds_f   = preds[mask]
    actuals_f = actuals[mask]

    rel_error = np.abs(preds_f - actuals_f) / actuals_f
    print(f"Mean relative error:   {rel_error.mean()*100:.2f}%")
    print(f"Median relative error: {np.median(rel_error)*100:.2f}%")
    print(f"Max relative error:    {rel_error.max()*100:.2f}%")
    print(f"Samples evaluated:     {mask.sum()}")