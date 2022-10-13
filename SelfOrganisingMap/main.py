import os
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt

from map import SelfOrganisingMap


color_map = {
    0.0: (1.0, 0.0, 0.0, 1.0),    # Red
    1.0: (0.0, 1.0, 0.0, 1.0),    # Green
    2.0: (0.0, 0.0, 1.0, 1.0),    # Blue
}


def read_data(file_name: str) -> np.ndarray:
    """
    Read data from csv file
    """
    return pd.read_csv(file_name, delimiter=',', header=None).values


def preprocessing(data: np.ndarray) -> np.ndarray:
    """
    Preprocess the data by normalising it to the range [0, 1]
    """
    return data / np.max(data)


def plot_colorcoded_data(points: list[tuple[int, int]], labels: np.ndarray, shape: tuple[int, int], ax):
    """
    Plot a time series in 3D
    """
    img = np.zeros((*shape, 4))
    for label, (x, y) in zip(labels, points):
        img[x, y] = color_map[label[0]]
    ax.imshow(img)


def train(som: SelfOrganisingMap, data: np.ndarray, epochs: int):
    """
    Train a self organasing map on the given data
    """
    lr0 = 0.1
    sigma0 = 10
    lr_decay = 0.01
    sigma_decay = 0.05

    batch_size, _ = data.shape
 
    for epoch in trange(epochs):
        lr = lr0 * np.exp(-lr_decay * epoch)
        sigma = sigma0 * np.exp(-sigma_decay * epoch)
        for _ in range(batch_size):
            x = data[np.random.randint(batch_size),:]
            som.train(x, lr, sigma)
    return som


def main():
    output_shape = (40, 40)
    epochs = 100

    # Load and preprocess data
    iris_data = read_data('iris-data.csv')
    iris_labels = read_data('iris-labels.csv')
    iris_data = preprocessing(iris_data)

    som = SelfOrganisingMap(4, output_shape)

    # Map with initial weights
    random_iris_map = np.array([som.run(x) for x in iris_data])

    # Train and get a new and better map
    train(som, iris_data, epochs)
    iris_map = np.array([som.run(x) for x in iris_data])

    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_colorcoded_data(random_iris_map, iris_labels, output_shape, ax1)
    plot_colorcoded_data(iris_map, iris_labels, output_shape, ax2)
    plt.show()

if __name__ == '__main__':
    # To read data correctly
    os.chdir(os.path.dirname(__file__))

    main()