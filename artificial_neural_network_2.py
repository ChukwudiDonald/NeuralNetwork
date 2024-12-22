import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import matplotlib_inline.backend_inline as mpl_inline

# Set the Matplotlib output format to SVG for better quality visuals in Jupyter notebooks
mpl_inline.set_matplotlib_formats('svg')  # New method

# Function to build and train the model
def buildAndTrainTheModel(x, y):
    # Define a simple neural network model using nn.Sequential
    ANNreg = nn.Sequential(
        nn.Linear(1, 1),  # First layer: Linear transformation (1 input, 1 output)
        nn.ReLU(),        # ReLU activation function to introduce non-linearity
        nn.Linear(1, 1)   # Second layer: Linear transformation (1 input, 1 output)
    )

    # Define the Mean Squared Error loss function
    lossfun = nn.MSELoss()

    # Define the optimizer using Stochastic Gradient Descent (SGD)
    optmizer = torch.optim.SGD(ANNreg.parameters(), lr=0.05)

    # Set the number of training epochs
    no_epochs = 250

    # Initialize a tensor to store loss values over epochs
    losses = torch.zeros(no_epochs)

    # Training loop
    for epoch in range(no_epochs):
        # Perform a forward pass: calculate predictions for current x
        yHat = ANNreg(x)

        # Compute the loss (difference between predicted and actual values)
        loss = lossfun(yHat, y)
        losses[epoch] = loss

        # Zero out the gradients before performing backpropagation
        optmizer.zero_grad()

        # Perform backpropagation to calculate gradients
        loss.backward()

        # Update model parameters (weights) based on gradients
        optmizer.step()

    # After training, obtain the model's predictions for the input data
    predictions = ANNreg(x)

    return predictions, losses

# Function to create synthetic data for training
def createTheData(m):
    # Generate random data for inputs (x) and outputs (y)
    N = 50  # Number of data points
    x = torch.randn(50, 1)  # Generate N random numbers for inputs (normally distributed)
    y = m * x + torch.randn(N, 1) / 2  # Generate outputs based on x with added noise

    return x, y

# Create synthetic data using a slope factor of 0.8
x, y = createTheData(0.8)

# Train the model and get predictions and loss values
yHat, losses = buildAndTrainTheModel(x, y)

# Generate a range of slope values to test the model's performance
slopes = np.linspace(-2, 2, 21)

# Number of experiments to run for each slope value
numExps = 50

# Initialize a results array to store loss and correlation values
results = np.zeros((len(slopes), numExps, 2))

# Run experiments with different slope values and record results
for slope in range(len(slopes)):

    for N in range(numExps):
        # Generate new data for each slope and experiment
        x, y = createTheData(slopes[slope])

        # Train the model and get predictions and loss values
        yhat, losses = buildAndTrainTheModel(x, y)

        # Store the final loss after training
        results[slope, N, 0] = losses[-1]

        # Calculate and store the correlation coefficient between real data and predictions
        results[slope, N, 1] = np.corrcoef(y.T, yHat.detach().T)[0, 1]

# Handle potential NaN values in the results array
results[np.isnan(results)] = 0

# Plotting results
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Plot the average loss across experiments for each slope
ax[0].plot(slopes, np.mean(results[:, :, 0], axis=1), 'ko-', markerfacecolor='w', markersize=10)
ax[0].set_xlabel('Slope')  # X-axis label
ax[0].set_title('Loss')    # Plot title

# Plot the average correlation between real data and predictions for each slope
ax[1].plot(slopes, np.mean(results[:, :, 1], axis=1), 'ms-', markerfacecolor='w', markersize=10)
ax[1].set_xlabel('Slope')  # X-axis label
ax[1].set_ylabel('Real-predicted correlation')  # Y-axis label
ax[1].set_title("Model performance")  # Plot title

# Show the plots
plt.show()
