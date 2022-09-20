import numpy as np


class OneLayerPerceptron:
    """
    Class for a Neural Network with a single hidden layer, 2 inputs and one output
    """

    def __init__(self, M: int) -> None:
        """
        M: Number of hidden neurons
        """
        # Initialze weights and bias
        self.w1 = np.random.normal(0, 1, size=(M, 2))
        self.t1 = np.zeros((M, 1))
        self.w2 = np.random.normal(0, 1, size=(1, M))
        self.t2 = np.zeros((1, 1))

        # Save computations for the backpropagation
        # b_i is the local field for layer i
        self.b1 = None
        self.b2 = None
        # x is the input
        self.x = None
        # V is the hidden layer after activatoin
        self.V = None
        # O is the output from the network
        self.O = None
        # d_i is the error for layer i
        self.d1 = None
        self.d2 = None

    def run(self, x: np.ndarray) -> np.ndarray:
        """
        Run the model with the input x
        """
        self.x = x
        self.b1 = self.w1 @ self.x - self.t1
        self.V = np.tanh(self.b1)
        self.b2 = self.w2 @ self.V - self.t2
        self.O = np.tanh(self.b2)

        return self.O

    def train_batch(self, inputs: list[np.ndarray], targets: list[np.ndarray]) -> float:
        """
        Train the network on a batch of data and return the total error over the data
        """
        error = 0
        for x, t in zip(inputs, targets):
            output = self.run(x)
            error += 0.5 * (t - output)**2

    def backpropagation(self) -> None:
        """
        Backpropagate through the network to calculate the error for each layer
        """
        self.d2 = None



    