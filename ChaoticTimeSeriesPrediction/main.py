import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from reservoir import ReservoirComputer


def read_data(file_name: str) -> np.ndarray:
    """
    Read data from csv file
    """
    return pd.read_csv(file_name, delimiter=',', header=None).values


def plot_3D_time_series(X: np.ndarray, color: str = 'b', ax=None):
    """
    Plot a time series in 3D
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    x, y, z = X
    ax.plot3D(x, y, z, c=color)
    return ax


def main():
    reservoirComputer = ReservoirComputer(3, 500, 3)

    # Load training data and train the reservoir computer
    X_train = read_data('training-set.csv')
    reservoirComputer.train(X=X_train[:,:-1], Y=X_train[:,1:], k=0.001)

    
    # Load test data and generate the continuation
    X_test = read_data('test-set-7.csv')
    O = reservoirComputer.dynamics(X_test, T=500)

    # Plot the solution
    ax = plot_3D_time_series(X_test, color='b')
    plot_3D_time_series(O, color='r', ax=ax)

    plt.show()

    # Save the y component of the result
    np.savetxt('prediction.csv', O[1,:], delimiter=',')


if __name__ == '__main__':
    # To read data correctly
    os.chdir(os.path.dirname(__file__))

    main()