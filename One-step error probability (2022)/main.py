import numpy as np
import random

from tqdm import tqdm


def sgn(x: np.ndarray) -> np.ndarray:
    """
    Perorm sgn(x) on the input. sgn(x) = +1 if x >= 0 and -1 if x < 0
    """
    sgn_value = np.sign(x)

    # Numpys sign function sets the output to 0 for x == 0. Correct for this
    sgn_value[sgn_value == 0] = 1
    return sgn_value


def random_pattern(N: int) -> np.ndarray:
    """
    Create a random pattern of N bits. The bits will be either +1 or -1 with 50% probability
    The pattern is stored in a [N x 1] matrix
    """
    return np.random.choice([-1, 1], size=(N, 1))


def hebbs_rule(patterns: list[np.ndarray]) -> np.ndarray:
    """
    Create the weight matrix W for a Hopfiled network using Hebb's rule. 
    This implementation sets the diagonal elements to zero

    Args:
        patterns: The patterns to store in the network
    
    Returns:
        A squrare matrix with a side lenght equal to the length of the patterns
    """
    # Initialize the weights
    N = patterns[0].shape[0]
    weights = np.zeros((N, N))

    # Store the patterns
    for x in patterns:
        weights += np.matmul(x, x.T)

    # Set the diagonal to 0
    np.fill_diagonal(weights, 0)

    # Return the weights after normalizing
    return weights / N


def asynchronous_update(state: np.ndarray, weights: np.ndarray, i: int) -> np.ndarray:
    """
    Update the state vector asynchronously, e.g. update only index i int the state

    Args:
        state: The current state vector
        weights: The weights connecting the neurons in the state
        i: Index in the state vextor to updtate
    
    Returns:
        The asynchronously updated state
    """
    state_tmp = np.copy(state)
    state_tmp[i, :] = sgn(np.matmul(weights[i, :], state_tmp))
    return state_tmp


def error_trial(N: int, p: int) -> int:
    """
    Do one independent trial too estimate the one-step error probability. 

    Args:
        N: Number of bits in the patterns
        p: Number of patterns
    
    Returns:
        Number of errors (0 or 1)
    """
    # Sample p random patterns with N bits
    patterns = [random_pattern(N) for _ in range(p)]
    # Get the weights with Hebb's rule
    W = hebbs_rule(patterns)
    # Select a random pattern to feed the network
    s0 = random.choice(patterns)
    # Select an random index to update
    i = np.random.randint(N)
    # Update the state of the network and compare check if an error has occured
    s1 = asynchronous_update(s0, W, i)
    if s0[i] == s1[i]:
        return 0  # No error
    return 1  # Error


if __name__ == '__main__':
    N = 120
    Ps = [12,24,48,70,100,120]
    samples = 100_000
    p_error = []
    for p in Ps:
        errors = 0
        print(f"\nEstimating one-step error probability with N = {N} and p = {p}")
        for _ in tqdm(range(samples)):
            errors += error_trial(N, p)
        one_step_error_prob = errors / samples
        print(f"One-step error probability estimeted to {one_step_error_prob: .04f} for N = {N} and p = {p}")
        p_error.append(one_step_error_prob)
    print(f"Single-step error probability: {p_error}")
    print(f"For the N = {N} and p = {Ps}")
        


