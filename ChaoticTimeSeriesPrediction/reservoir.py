import numpy as np


class ReservoirComputer:
    """
    A reservoir computer
    """

    def __init__(self, inputs: int, reservoir_size: int, outputs: int) -> None:
        self.w_in = np.random.normal(loc=0, scale=np.sqrt(0.002), size=(reservoir_size, inputs))
        self.w = np.random.normal(loc=0, scale=np.sqrt(0.004), size=(reservoir_size, reservoir_size))
        self.w_out = np.empty((outputs, reservoir_size))

        self.reservoir_size = reservoir_size
        self.outputs = outputs

    def train(self, X: np.ndarray, Y: np.ndarray, k: float) -> None:
        """
        Train the reservoir computers output weights using ridge regression

        X: [inputs, T] matrix
        Y: [outputs, T] matrix
        """
        # Generate the hidden reservoir for all timesteps in the training data
        T = X.shape[1]
        reservoir = np.zeros((self.reservoir_size, T + 1))
        for t in range(0, T):
            reservoir[:,t + 1] = np.tanh(self.w @ reservoir[:,t] + self.w_in @ X[:,t])

        # Remove initial transient
        reservoir = reservoir[:,501:]
        Y = Y[:,500:]

        # Calculate the output weights with ridge regression
        # self.w_out = np.linalg.inv(reservoir.T @ reservoir + k * np.eye(self.reservoir_size)) @ reservoir.T @ Y
        self.w_out = Y @ reservoir.T @ np.linalg.inv(reservoir @ reservoir.T + k * np.eye(self.reservoir_size))

    def dynamics(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Generate the dynamics of the system

        X: [inputs, T] matrix
        T: Timesteps to take after the initial input stops
        """
        # Set up the initial memory
        T1 = X.shape[1]
        reservoir = np.zeros((self.reservoir_size,))
        for t in range(T1):
            reservoir = np.tanh(self.w @ reservoir + self.w_in @ X[:,t])

        # Keep iterating the dynamics
        O = np.empty((self.outputs, T))
        for t in range(T):
            O[:,t] = self.w_out @ reservoir
            reservoir = np.tanh(self.w @ reservoir + self.w_in @ O[:,t])

        return O
