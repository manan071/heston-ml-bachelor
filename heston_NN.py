import numpy as np
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time 

# Load the generated data
data = np.loadtxt("heston_data.txt", delimiter=",")

# Set GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    X = data[:,:-1]
    y = data[:,-1]

    # Split the data into training set, testing set and validation set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

    # Standardize the features and target variable
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1,1))

    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1,1))

    X_val = scaler_X.transform(X_val)
    y_val = scaler_y.transform(y_val.reshape(-1,1))

    # Create DataLoaders for mini-batch training
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # Create the model
    model = HestonNN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)

    # Convert test data to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    # Convert validation data to PyTorch tensors
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    # Train the model
    best_loss = float('inf')
    patience_counter = 0

    start_time = time.time()

    for epoch in range(150):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        # Evaluate the model on the validation set 
        model.eval()

        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val)

        scheduler.step(val_loss)   

        if epoch % 10 == 0:
            training_log = (
                f"Epoch {epoch}, "
                f"Loss: {epoch_loss / len(train_loader.dataset)}, "
                f"Validation Loss: {val_loss.item()}, "
                f"LR: {optimizer.param_groups[0]['lr']}"
            )
            print(training_log)

            # Write training log to file
            with open("training_log.txt", "a") as f:
                f.write(training_log + "\n")

        # Early stopping based on validation loss and save the best model
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0

            # Save the model and scalers
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_X': scaler_X,
                'scaler_y': scaler_y
            }, 'heston_nn.pth')
        else:
            patience_counter += 1
            if patience_counter >= 50:  # Early stopping after 50 epochs without improvement
                print(f"Early stopping triggered at epoch {epoch}")
                break

    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")

    print()

    # Load the best model for evaluation
    checkpoint = torch.load('heston_nn.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Best model loaded for evaluation.")

    # Test the model with specific parameters
    test_input_standardized = scaler_X.transform([[np.log(100/100), 0.5, 5.0, 0.05, 0.5, -0.8, 0.05, 0.03, 0.02]])
    test_input = torch.tensor(test_input_standardized, dtype=torch.float32).to(device)

    with torch.no_grad():
        nn_price  = model(test_input)
        nn_price = scaler_y.inverse_transform(nn_price.cpu().numpy())

    print(nn_price[0][0]*100)

    print()

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