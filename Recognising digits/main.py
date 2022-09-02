import numpy as np


# Patterns to store
x1 = np.array([ [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] ])
x2 = np.array([ [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1] ])
x3 = np.array([ [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1] ])
x4 = np.array([ [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1] ])
x5 = np.array([ [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1] ])

# Patterns to feed
s01 = np.array([[1, 1, -1, -1, -1, -1, -1, -1, 1, 1], [1, 1, -1, -1, -1, -1, -1, -1, -1, 1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1]])
s02 = np.array([[1, -1, -1, -1, -1, 1, -1, -1, -1, -1], [1, -1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, 1, 1, 1, -1, 1, 1, -1, -1], [1, 1, 1, 1, -1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1, 1, 1, 1, -1], [1, -1, 1, 1, 1, -1, 1, 1, -1, -1], [1, -1, -1, 1, 1, -1, 1, -1, -1, -1], [1, -1, -1, -1, -1, 1, -1, -1, -1, -1]])
s03 = np.array([[1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, 1, -1, 1, -1, 1, -1, -1], [1, 1, -1, 1, -1, 1, -1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1]])

def sgn(x: np.ndarray) -> np.ndarray:
    """
    Perorm sgn(x) on the input. sgn(x) = +1 if x >= 0 and -1 if x < 0
    """
    sgn_value = np.sign(x)
    # Numpys sign function sets the output to 0 for x == 0. Correct for this
    sgn_value[sgn_value == 0] = 1
    return sgn_value


def hebbs_rule(patterns: list[np.ndarray], zero_diagonal: bool = True) -> np.ndarray:
    """
    Create the weight matrix W for a Hopfiled network using Hebb's rule. 
    This implementation sets the diagonal elements to zero

    Args:
        patterns: The patterns to store in the network
        zero_diagonal: If the diagonal elements in the weight matrix should be set to zero
    
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
    if zero_diagonal:
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


def update_until_steady_state(s0: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Update a initial state s0 asynchronously with the weights W until a steady state is reached
    """
    prev = s0
    next = s0
    N = s0.shape[0]
    while True:
        for i in range(N):
            next = asynchronous_update(next, W, i)
        # Return the result when a steady state is reached
        if np.all(prev == next):
            return next
        prev = next


def decode_state(s: np.ndarray, stored_patterns: list[np.ndarray]) -> int:
    """
    Decode the state stored in s. 
    If it matches the pattern at index i in the list of stored patterns return i+1.
    If it matches the inverse of the pattern i return -(i+1).
    If no pattern matches, return len(stored_patterns) + 1
    """
    for i, x in enumerate(stored_patterns):
        if np.all(s == x):
            return i + 1
        if np.all(s == -x):
            return -(i + 1)
    return len(stored_patterns) + 1

if __name__ == '__main__':
    # Setup
    patterns = [np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1)), np.reshape(x3, (-1, 1)), np.reshape(x4, (-1, 1)), np.reshape(x5, (-1, 1))]
    W = hebbs_rule(patterns, zero_diagonal=True)

    for s in [s01, s02, s03]:
        initial_state = np.reshape(s, (-1, 1))

        # Run the network until it converges
        steady_state = update_until_steady_state(initial_state, W)

        # Decode the output
        print(f"\nThe pattern converged to the state {decode_state(steady_state, patterns)}")
        # Reshape to correct format again
        steady_state.shape = s01.shape
        print(steady_state)
