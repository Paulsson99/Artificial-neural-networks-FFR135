import numpy as np
from tqdm import tqdm


def sgn(x: float) -> int:
    """
    Perorm sgn(x) on the input. sgn(x) = +1 if x >= 0 and -1 if x < 0
    """
    if x < 0:
        return -1
    return 1


def generate_truth_table(n: int) -> np.ndarray:
    """
    Generate a truth table with n inputs
    """
    # Create an array with all the integers [0, 2^n-1]
    d = np.arange(2**n)
    # Array with all powers of 2, [2^0, 2^1, ..., 2^(n-1)]
    powers = 1 << np.arange(n)
    # Bitwise and to extract the the binary representation
    b = d[:, None] & powers 
    # Replace all numbers > 0 with 1
    return (b > 0).astype(int)


def hash_boolean_func(targets: np.ndarray) -> int:
    """
    Hash the boolen function to give each function a unique number identification
    The algorithm used will be

    h = x_1*2^0 + x_2*2^1 + ... + x_n*2^n

    h is the hash value, x_i is the i:th output of a boolean function
    To make things work proparly -1 will be mapped to 0 first
    """
    h = 0
    for i, x in enumerate(targets):
        if x == 1:
            h += 2**i
    return h


def run_model(inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> int:
    """
    Run the model with the given weihgts and bias
    """
    return sgn(np.dot(weights, inputs) - bias)


def train_model(inputs: np.ndarray, targets: int, generations: int, learning_rate: float) -> tuple[np.ndarray, float]:
    """
    Initialaze random weights and a bias and train the model
    """
    n = inputs.shape[1]
    weights = np.random.normal(0, 1 / n, size=n)
    bias = 0

    for _ in range(generations):
        for x, t in zip(inputs, targets):
            output = run_model(x, weights, bias)
            weights += learning_rate * (t - output) * x
            bias -= learning_rate * (t - output)

    return weights, bias


def main():
    N = 10_000
    dims = [2, 3, 4, 5]
    lr = 0.05
    generations = 20

    for n in dims:
        print(f"Finding the number of linerarly separable boolean functions for n={n}")
        sampled_functions = {}
        truth_table = generate_truth_table(n)

        for _ in tqdm(range(N)):
            targets = np.random.choice([-1, 1], size=2**n)
            hash = hash_boolean_func(targets)
            
            trained_weights, trained_bias = train_model(truth_table, targets, generations, lr)

            for x, t in zip(truth_table, targets):
                if not run_model(x, trained_weights, trained_bias) == t:
                    sampled_functions[hash] = 0
                    break
            else:
                sampled_functions[hash] = 1
        
        linerarly_separable_functions = sum(sampled_functions.values())
        print(f"Number of linearly separable boolean functions found for n={n} is {linerarly_separable_functions}")


if __name__ == '__main__':
    main()

    #   n | sum | actual    | total         | %
    #   2 | 14  | 14        | 16            | 0.875
    #   3 | 104 | 104       | 256           | 0.406
    #   4 | 266 | 1882      | 65536         | 0.029
    #   5 | 1   | 94572     | 4294967296    | 0.000022019259632