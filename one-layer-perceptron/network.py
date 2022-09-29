import numpy as np


class TangensHyperbolicus:
    """
    Perform tanh(ax)
    """

    def __init__(self, a: float) -> None:
        self.a = a

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.a * x)

    def derivative(self, f: np.ndarray) -> np.ndarray:
        return self.a * (1 - f**2)


class Network:
    """
    A deep neural network
    It uses the tanh function for activation
    """

    def __init__(self, layers: list[int]):
        self.layers = len(layers) - 1
        self.weights: list[np.ndarray] = [] 
        self.bias: list[np.ndarray] = []
        self.activation = TangensHyperbolicus(1)

        # Variables to save for gradient decent
        self.V = [None] * (self.layers + 1)
        self.local_fields: list[np.ndarray] = [None] * self.layers

        for n, m in zip(layers[:-1], layers[1:]):
            self.weights.append(np.random.random((n, m)) * 2 - 1)
            self.bias.append(np.zeros((m,)))

    @property
    def size(self) -> int:
        """
        Return the total size of the network (number of tunable parameters)
        """
        params = 0
        for w, b in zip(self.weights, self.bias):
            params += w.size + b.size
        return params

    def run(self, x: np.ndarray) -> np.ndarray:
        """
        Feed input x to the network. x should be a 2D numpy array with the shape (BATCH_SIZE, N),
        were N is the number of inputs in a pattern and BATCH_SIZE is the number of patterns. 
        
        NOTE: This implies that we are working with row vectors as input so our weight matrix will be transposed!!!
        """
        self.V[0] = x
        for i in range(self.layers):
            w, b = self.weights[i], self.bias[i]
            local_field = x @ w - b
            self.local_fields[i] = local_field
            x = self.activation(local_field)
            self.V[i + 1] = x
        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        batch_size = x.shape[0]
        output = self.run(x)
        # Reshape the target to match the output
        t = t.reshape(output.shape)
        return 0.5 * np.sum((t - output)**2) / batch_size

    def classification_error(self, x: np.ndarray, t: np.ndarray) -> float:
        batch_size = x.shape[0]
        output = self.run(x)
        t = t.reshape(output.shape)
        return np.sum(np.abs(np.sign(output) - t)) / (2 * batch_size)

    def backpropagation(self, x: np.ndarray, t: np.ndarray, lr: float):
        """
        Perform backpropagation with some training data
        """
        # Rescale the learning rate for the batch size

        if len(x.shape) < 2:
            x = np.expand_dims(x, axis=0)

        batch_size = x.shape[0]
        lr /= batch_size
        
        outputs = self.run(x)
        dw = [None] * self.layers
        db = [None] * self.layers

        # Reshape the target to match the output
        t = t.reshape(outputs.shape)
        if len(t.shape) < 2:
            t = np.expand_dims(t, axis=0)

        error = (t - outputs) * self.activation.derivative(self.V[-1])
        dw[-1] = lr * np.sum(self.V[-2][:,:,np.newaxis] @ error[:,np.newaxis,:], axis=0)
        db[-1] = -lr * np.sum(error, axis=0)

        for l in range(self.layers - 1, 0, -1):
            error = error @ self.weights[l].T * self.activation.derivative(self.V[l])
            dw[l - 1] = lr * np.sum(self.V[l - 1][:,:,np.newaxis] @ error[:,np.newaxis,:], axis=0)
            db[l - 1] = -lr * np.sum(error, axis=0)

        # Apply the calculated gradient for the weights and biases
        for i in range(self.layers):
            self.weights[i] += dw[i]
            self.bias[i] += db[i]
