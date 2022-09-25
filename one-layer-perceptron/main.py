import os
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt

from network import Network


# Constants
M = 100
EPOCHS = 1000
BATCH_SIZE = 10
LEARNING_RATE = 0.3


def read_data() -> tuple[np.ndarray]:
    """
    Read the training and validation data from the csv files
    """
    training_data = pd.read_csv('training_set.csv', delimiter=',', header=None).values
    validation_data = pd.read_csv('validation_set.csv', delimiter=',', header=None).values

    training_x, training_t = training_data[:,:2], training_data[:,-1]
    validation_x, validation_t = validation_data[:,:2], validation_data[:,-1]

    # Add an extra dimension to the data so it will be 2D when iterating over it
    return training_x, training_t, validation_x, validation_t


def preprocess_data(data: np.ndarray) -> np.ndarray:
    """
    Shift and scale the data so the mean becomes 0 and the variance 1
    Return the required shift and scale factor
    """
    shift = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    return shift, 1 / np.sqrt(variance)


def plot_data():
    """
    The data is 2d so lets plot it to estimate how many hidden neurons are needed
    """
    training_x, training_t, validation_x, validation_t = read_data()
    shift, scale = preprocess_data(training_x)
    training_x = (training_x - shift) * scale
    validation_x = (validation_x - shift) * scale

    print(f'Shifting data with {shift}. Scaling with {scale}')

    coord = np.concatenate((training_x, validation_x))
    target = np.concatenate((training_t, validation_t))
    colors = ['blue' if t > 0 else 'red' for t in target]
    plt.scatter(coord[:,0], coord[:,1], c=colors)
    plt.show()

def update_line(line, x, y):
    line.set_xdata(np.append(line.get_xdata(), x))
    line.set_ydata(np.append(line.get_ydata(), y))
    line.get_figure().canvas.draw()


def main():
    # Preprocessing
    training_x, training_t, validation_x, validation_t = read_data()
    shift, scale = preprocess_data(training_x)
    training_x = (training_x - shift) * scale
    validation_x = (validation_x - shift) * scale

    print(f'Shifting data with {shift}. Scaling with {scale}')
    

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots()
    training_loss_line, = ax.plot([], [], '-', label='Training loss')
    validation_loss_line, = ax.plot([], [], label='Validation loss')
    training_error_line, = ax.plot([], [], '-', label='Training error')
    validation_error_line, = ax.plot([], [], label='Validation error')
    ax.legend()
    ax.set_xlim(1, EPOCHS)
    ax.set_ylim(0, 0.5)

    # Setup the network
    nn = Network([2, M, 1])
    best_validation_error = np.inf
    best_w = None
    best_t = None

    training_samples = training_x.shape[0]

    for epoch in trange(EPOCHS):

        # Train
        for _ in range(training_samples // BATCH_SIZE):
            batch_index = np.random.choice(training_samples, BATCH_SIZE, replace=False)
            nn.backpropagation(training_x[batch_index], training_t[batch_index], lr=LEARNING_RATE)

        # Calculate loss and classification error
        training_loss = nn.loss(training_x, training_t)
        validation_loss = nn.loss(validation_x, validation_t)
        training_error = nn.classification_error(training_x, training_t)
        validation_error = nn.classification_error(training_x, training_t)

        # Save the weights that gives the smalest validation error
        if validation_error < best_validation_error:
            best_validation_error = validation_error
            best_w = [w.T.copy() for w in nn.weights]
            best_t = [t.copy() for t in nn.bias]

        # Quit the program if the figure is closed
        if not plt.fignum_exists(fig.number):
            print("Training aborted")
            break
        
        # Update the figure
        update_line(training_error_line, epoch + 1, training_error)
        update_line(validation_error_line, epoch + 1, validation_error)
        update_line(training_loss_line, epoch + 1, training_loss)
        update_line(validation_loss_line, epoch + 1, validation_loss)
        plt.pause(0.1)

    # Print best result
    print(f"The best network had a classification error of {best_validation_error:.2%}")

    # Save the weights and bias
    for i, w in enumerate(best_w):
        np.savetxt(f"w{i + 1}.csv", w.reshape((M, -1)), delimiter=',')
    for i, t in enumerate(best_t):
        np.savetxt(f"t{i + 1}.csv", t[:,np.newaxis], delimiter=',')

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    # To read data correctly
    os.chdir(os.path.dirname(__file__))

    # plot_data()
    main()