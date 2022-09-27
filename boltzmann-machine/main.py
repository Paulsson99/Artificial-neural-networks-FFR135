from calendar import EPOCH
import numpy as np

from boltzmann import BoltzmannMachine


# Constants
EPOCHS = 1
BATCH_SIZE = 30
LEARNING_RATE = 0.05
CD_K = 200


def train_XOR(M: int) -> float:
    """
    Train a boltzmann machine with M hidden neurons on the XOR problem.
    Estimate and return the Kullback-Leibler divergence
    """
    p_data = np.array(
        [[-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [1, 1, 1]]
    )
    bm = BoltzmannMachine(visable=3, hidden=M)
    for epoch in range(EPOCHS):
        batch_index = np.random.randint(4, size=BATCH_SIZE)
        bm.train(p_data[batch_index], k=CD_K, lr=LEARNING_RATE)

    # Sample the data distrubution for the model
    for _ in range(1000):
        bm.update() # Remove initial transient

    frequenzy_table = {i: 0 for i in range(8)}
    for _ in range(10_000):
        bm.update()
        frequenzy_table[hash_state(bm.visable)] += 1

    # Sample the Kullback-Leibler divergence
    p_model = [frequenzy_table[hash_state(p)] / 10_000 for p in p_data]

    print(p_model)

    D_KL = sum([1/4 * (np.log(1/4) - np.log(p_f)) for p_f in p_model])

    return D_KL


def hash_state(state: np.ndarray):
    """
    Hash a network state
    """
    h = 0
    for i, x in enumerate(state):
        if x > 0:
            h += 2**i
    return h


def main():
    M = [1, 2, 4, 8]

    for m in M:
        KullbackLeibler = train_XOR(m)


if __name__ == '__main__':
    main()
