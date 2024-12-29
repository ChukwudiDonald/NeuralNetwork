import matplotlib.pyplot as plt
from AnnClass import AnnClass
from data import file_paths
import numpy as np
import helpers
import torch

df = helpers.read_csv_with_default(
                                    file_path=file_paths[3],
                                    delimiter=';'
                                   )

N = 10000
tolerance = 0.001
data_column = [
               "Open",
               "High",
               "Low",
               "Close",
               "Volume"
               ]
x, y = helpers.create_input_output(df[[data_column[0]]].values, end_index=N)


M = AnnClass(
              epochs=10_000,
              learning_rate=0.0001
             )

M.train(x,y)

predictions = M.evaluate_and_print_loss(x, y)

test_loss = (predictions - y).pow(2).mean() # Calculate the test loss (Mean Squared Error) on the predictions

# Plotting the loss during training
plt.plot(M.losses.detach(), 'o', markerfacecolor='w', linewidth=0.1)  # Plotting losses with white-filled markers
plt.plot(M.num_epoch, test_loss.detach(), "ro")  # Plotting test loss as red circles
plt.xlabel("Epoch")  # Label for the x-axis (Epoch)
plt.ylabel("Loss")  # Label for the y-axis (Loss)
plt.title("Final loss = %g" % test_loss.item())  # Displaying the final test loss in the title
plt.show()  # Show the loss plot

# Plotting real data vs predictions
plt.plot(x, y, 'bo', label="RealData")  # Plotting real data as blue circles
plt.plot(x, predictions.detach(), 'rs', label='Predictions')  # Plotting predictions as red squares
plt.title(f"prediction-data r={np.corrcoef(y.T, predictions.detach().T)[0, 1]: .2f}")  # Display correlation coefficient in title
plt.legend()  # Show legend for the real data and predictions
plt.show()  # Show the plot for real data vs predictions

# Plotting real data vs predictions over time
time_sequence = np.arange(len(y))  # Generating a time sequence based on the length of y
plt.plot(time_sequence, y, 'b-', label="RealData")  # Plotting real data as a blue line
plt.plot(time_sequence, predictions.detach(), 'r-', label='Predictions')  # Plotting predictions as a red line
plt.title(f"prediction-data r={np.corrcoef(y.T, predictions.detach().T)[0, 1]: .2f}")  # Display correlation coefficient in title
plt.legend()  # Show legend for the real data and predictions over time
plt.show()  # Show the time-based plot for real data vs predictions

for loss in M.loss_per_cycle:
    print(loss)
