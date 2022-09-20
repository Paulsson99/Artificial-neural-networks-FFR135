import numpy as np

from network import OneLayerPerceptron


nn = OneLayerPerceptron(4)
print(nn.run(np.array([[1], [1]])))
