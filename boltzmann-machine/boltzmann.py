import numpy as np


class BoltzmannMachine:
    """
    A class implementing a restricted Boltzann Machine
    """

    def __init__(self, visable: int, hidden: int) -> None:
        self.w = np.random.random((hidden, visable))
        self.tv = np.zeros(visable)
        self.th = np.zeros(hidden)

        self.visable = np.zeros(visable)
        self.hidden = np.zeros(hidden)

    def initialize_network(self, x: np.ndarray) -> None:
        """
        Initialize the network states
        """
        self.visable = x.copy()
        self.hidden = self.mcCullochPittsDynamics(self.visable, self.w, self.th)

    def update(self):
        """
        Update the network
        """
        self.visable = self.mcCullochPittsDynamics(self.hidden, self.w.T, self.tv)
        self.hidden = self.mcCullochPittsDynamics(self.visable, self.w, self.th)

    @staticmethod
    def mcCullochPittsDynamics(x: np.ndarray, W: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Calculate the output with the McCulloch-Pitts dynamics
        """
        b = W @ x - t
        p = 1 / (1 + np.exp(-2 * b))
        return 2 * np.random.binomial(1, p) - 1

    def train(self, pattern_batch: np.ndarray, k: int, lr: float) -> None:
        """
        Train the network for k generations on patterns to approximate the distrubution of p_data
        """
        lr = lr / pattern_batch.shape[0]
        dw = np.zeros_like(self.w)
        dtv = np.zeros_like(self.tv)
        dth = np.zeros_like(self.th)
        for x in pattern_batch:
            self.initialize_network(x)
            for _ in range(k):
                self.update()
            
            bh_0 = self.w @ x - self.th
            bh_k = self.w @ self.visable - self.th
            dw += np.outer(np.tanh(bh_0), x) - np.outer(np.tanh(bh_k), self.visable)
            dtv += -(x - self.visable)
            dth += -(np.tanh(bh_0) - np.tanh(bh_k))
        
        # Apply the gradients
        self.w += lr * dw
        self.tv += lr * dtv
        self.th += lr * dth
