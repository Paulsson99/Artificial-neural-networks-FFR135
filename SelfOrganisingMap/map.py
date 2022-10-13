import numpy as np


class SelfOrganisingMap:
    """
    A self organinsing map
    """

    def __init__(self, inputs: int, output_shape: tuple[int, int]) -> None:
        self.w = np.random.random((*output_shape, inputs))
        self.output_shape = output_shape

    def run(self, x: np.ndarray) -> tuple[int, int]:
        """
        Map the input x to the location of the best neuron
        """
        distance_sq = np.sum((self.w - x)**2, axis=-1)
        return np.unravel_index(np.argmin(distance_sq), self.output_shape)

    def h(self, i: tuple[int, int], i0: tuple[int, int], sigma: float) -> float:
        """
        Calculate the neighbourhood function
        """
        r_i = np.array(i)
        r_i0 = np.array(i0)
        return np.exp(-np.sum( (r_i - r_i0)**2 ) / (2 * sigma**2))

    def train(self, x: np.ndarray, lr: float, sigma: float) -> None:
        """
        Train the weights for one input x
        """
        i0 = self.run(x)
        dw = np.empty_like(self.w)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                dw[i,j,:] = lr * self.h((i, j), i0, sigma) * (x - self.w[i,j,:])
        self.w += dw
