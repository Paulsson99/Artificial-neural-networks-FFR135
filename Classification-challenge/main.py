import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from network import Network


EPOCHS = 100
BATCH_SIZE = 100
LEARNING_RATE = 0.05


def load_mnist_data_to_classify() -> np.ndarray:
    """
    Load the data downloaded from CANVAS
    """
    return np.moveaxis(np.load('xTest2.npy'), -1, 0).reshape(10_000, 784)


def load_training_data() -> tuple[tuple[np.ndarray]]:
    """
    Load the training data
    """
    with np.load('mnist.npz') as f:
        x_train, y_train = f['x_train'].reshape(60_000, 784), f['y_train']
        x_test, y_test = f['x_test'].reshape(10_000, 784), f['y_test']
        return (x_train, vectorize_target(y_train), y_train), (x_test, vectorize_target(y_test), y_test)


def vectorize_target(y: np.ndarray) -> np.ndarray:
    """
    Convert the target to a one-hot vector
    """
    onehot = np.zeros((y.size, 10))
    onehot[np.arange(y.size), y] = 1
    return onehot


def display_digit(digit: np.ndarray) -> None:
    """
    Display a digit
    """
    plt.imshow(digit.reshape((28, 28)), cmap='gray')
    plt.show()


def update_line(line, x, y):
    line.set_xdata(np.append(line.get_xdata(), x))
    line.set_ydata(np.append(line.get_ydata(), y))
    line.get_figure().canvas.draw()


def main():
    # Load the training data
    (x_train, y_train, labels_train), (x_test, y_test, labels_test) = load_training_data()

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
    nn = Network([784, 200, 50, 10])

    training_samples = x_train.shape[0]

    for epoch in trange(EPOCHS):

        # Train
        for batch in range(training_samples // BATCH_SIZE):
            batch_index = np.random.choice(training_samples, BATCH_SIZE, replace=False)
            nn.backpropagation(x_train[batch_index], y_train[batch_index], lr=LEARNING_RATE)

        # Calculate loss and classification error
        training_loss = nn.loss(x_train, y_train)
        validation_loss = nn.loss(x_test, y_test)
        training_error = nn.classification_error(x_train, labels_train)
        validation_error = nn.classification_error(x_test, labels_test)

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
    
    # Classify the data from CANVAS
    x = load_mnist_data_to_classify()
    np.savetxt('classifications.csv', nn.classify(x), delimiter=',', fmt='%d')

    plt.ioff()
    plt.show()


def check_result():
    with open('classifications.csv', 'r') as f:
        x = load_mnist_data_to_classify()
        i = 0
        for line in f:
            print(line.strip())
            display_digit(x[i])
            if i > 20:
                break
            i += 1



if __name__ == '__main__':
    # To read data correctly
    os.chdir(os.path.dirname(__file__))

    # main()
    check_result()