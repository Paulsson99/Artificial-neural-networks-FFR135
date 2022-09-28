from operator import mod
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Turn of some logs from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Constants
EPOCHS = 5
BATCH_SIZE = 32


def load_mnist_data_to_classify() -> np.ndarray:
    """
    Load the data downloaded from CANVAS
    """
    return np.moveaxis(np.load('xTest2.npy'), -1, 0).reshape(-1, 28, 28, 1) / 255.0


def load_training_data(add_extra_dim: bool = False) -> tuple[tuple[np.ndarray]]:
    """
    Load the training data and normalize it
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


def display_digit(digit: np.ndarray) -> None:
    """
    Display a digit
    """
    plt.imshow(digit.reshape((28, 28)), cmap='gray')
    plt.show()


def flat_model(hidden_layers: list[int], dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Return a regular deep network
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    for l in hidden_layers:
        model.add(tf.keras.layers.Dense(l, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10))
    return model


def convolution_model(hidden_layers: list[int], dropout_rate: float = 0.2):
    """
    Return a network using convolution
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Flatten())
    for l in hidden_layers:
        model.add(tf.keras.layers.Dense(l, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10))
    return model


def main():
    # Load the training data
    (x_train, y_train), (x_test, y_test) = load_training_data()

    # Setup the model
    model = convolution_model([64])
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer='adam',
        loss=loss_func,
        metrics=['accuracy']
    )
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    # Train and evaluate
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, use_multiprocessing=True)
    model.evaluate(x_test, y_test, verbose=2)
    
    # Classify the data from CANVAS
    x = load_mnist_data_to_classify()
    predictions = probability_model(x)
    clasifications = tf.argmax(predictions, axis=1).numpy()
    np.savetxt('classifications.csv', clasifications, delimiter=',', fmt='%d')


def check_result(rows: int, cols: int):
    """
    Open a figure with rows*cols randomly chosen digits so the user can see if the network has learned anything
    """
    x = load_mnist_data_to_classify()
    classifications = np.loadtxt('classifications.csv', delimiter=',', dtype=int)
    rand_index = np.random.choice(classifications.shape[0], size=rows * cols, replace=False)
    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(cols, rows)
    for row in range(rows):
        for col in range(cols):
            ax = axes[row][col]
            i = rand_index[row * cols + col]
            ax.imshow(x[i], cmap='gray')
            ax.set_title(classifications[i])
            ax.set_axis_off()

    plt.subplots_adjust(
        left=0.0,
        right=1.0,
        top=0.95,
        bottom=0.05,
        wspace=0.1,
        hspace=0.8
    )
    plt.show()


if __name__ == '__main__':
    # To read data correctly
    os.chdir(os.path.dirname(__file__))

    main()
    check_result(rows=10, cols=18)