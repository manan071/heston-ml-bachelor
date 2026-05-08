import matplotlib.pyplot as plt

# Function to plot the training and validation loss
def plot_loss(log_history):
    plt.figure(figsize=(10, 5))
    plt.plot(log_history['train_loss'], label='Train Loss')