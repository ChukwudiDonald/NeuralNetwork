import pandas as pd
from data import file_paths
import torch
import matplotlib.pyplot as plt

def create_input_output(data_tensor, sequence_length):
    """
    Creates input-output pairs from a given data tensor by shifting the sequence.

    Parameters:
    - data_tensor (torch.Tensor): The input data for creating the sequences.
    - sequence_length (int): The length of the input sequence for `inputs`.

    Returns:
    - torch.Tensor: The input tensor `inputs`, which is the sequence up to the Nth element.
    - torch.Tensor: The output tensor `targets`, which is the sequence shifted by 1.
    """
    # Create the 'inputs' tensor by selecting the first 'sequence_length' elements and reshaping
    inputs = data_tensor[:sequence_length].view(-1, 1)

    # Create the 'targets' tensor by shifting 'inputs' by 1
    targets = inputs[1:].clone().float()  # Shift inputs by 1 to create targets

    # Remove the last element of 'inputs' to match the length of 'targets'
    inputs = inputs[:-1].clone().float()  # Remove the last value from inputs

    return inputs, targets


def group_data_correctly(data, tolerance=0.1):
    """
    Groups data based on the specified tolerance, comparing the decimal places.

    Parameters:
    - data (list): List of numerical values to be grouped.
    - tolerance (float): The precision level for grouping based on decimal places.

    Returns:
    - list: A list of grouped data.
    """
    # Sort the data first
    sorted_data = sorted(data)

    grouped = []
    current_group = [sorted_data[0]]  # Start the first group with the first number

    # Determine the number of decimal places to consider based on tolerance
    precision = len(str(tolerance).split('.')[1])  # Number of decimal places for tolerance

    for i in range(1, len(sorted_data)):
        previous_value = current_group[-1]
        current_value = sorted_data[i]

        # Extract the decimal part, and compare the first decimal places (e.g., 0.1 means tenth, 0.01 means hundredth, etc.)
        previous_decimal = str(previous_value).split('.')[1][:precision]
        current_decimal = str(current_value).split('.')[1][:precision]

        # If the first `precision` decimal digits are the same, group them
        if previous_decimal == current_decimal:
            current_group.append(current_value)
        else:
            grouped.append(current_group)
            current_group = [current_value]

    # Add the last group to the result
    grouped.append(current_group)

    return grouped


def average_of_sublists(lst):
    """
    Calculates the average of each sublist that has more than 3 elements.

    Parameters:
    - lst (list of lists): List of groups to calculate averages for.

    Returns:
    - list: A list of averages for sublists with more than 3 elements.
    """
    averages = []
    for sublist in lst:
        if len(sublist) >= 3:  # Only consider sublists with more than 3 elements
            avg = sum(sublist) / len(sublist)  # Calculate the average
            averages.append(avg)
    return averages


def get_important_levels(df_list, tolerance):
    """
    Groups the data and calculates the important levels (averages of subgroups).

    Parameters:
    - df_list (list): List of values to be grouped and averaged.
    - tolerance (float): The precision level for grouping.

    Returns:
    - list: The calculated important levels (averages of subgroups).
    """
    grouped_levels = group_data_correctly(df_list, tolerance)
    important_levels = average_of_sublists(grouped_levels)
    return important_levels


def plot_data(x, y, important_levels):
    """
    Plots the data points and the important horizontal lines on a graph.

    Parameters:
    - x (torch.Tensor): The input data for x-axis.
    - y (torch.Tensor): The output data for y-axis.
    - important_levels (list): List of important levels for horizontal lines.
    """
    fig, ax = plt.subplots()

    # Plot the data points (blue squares)
    ax.plot(x, y, 's', color='b', label='Data Points')

    # Plot horizontal lines at the important levels (red lines)
    for level in important_levels:
        ax.axhline(y=level, color='r', linestyle='--', label=f'Level {level}')

    # Add title, labels, and legend
    ax.set_title('Plot with Horizontal Lines')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Show the plot
    plt.show()


def main(df, N, tolerance):
    """
    Main function that ties everything together:
    - Prepares the data
    - Groups the data based on tolerance
    - Calculates the important levels
    - Plots the data and important levels

    Parameters:
    - df (DataFrame): The data frame containing the data.
    - N (int): The number of data points to use.
    - tolerance (float): The tolerance for grouping the data.
    """
    # Prepare the list for grouping (you already have the data prepared)
    df_list = df["Open"].values[:N].tolist()

    # Get the important levels by grouping and averaging
    important_levels = get_important_levels(df_list, tolerance)

    # Print the important levels
    print(important_levels)

    # Plot the data and important levels
    x, y = create_input_output(torch.tensor(df["Open"].values), N)  # You already have this part for preparing data
    plot_data(x, y, important_levels)


# Example usage
df = pd.read_csv(file_paths[6], encoding='utf-16', delimiter=';')
N = 2
tolerance = 0.001
for i in range(20):
    main(df, N+i, tolerance)
