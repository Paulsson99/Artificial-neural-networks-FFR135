from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from boltzmann import BoltzmannMachine


# Setup matplotlib
plt.rcParams['text.usetex'] = True

# Constants
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.1
CD_K = 100
AVERAGES = 20
SAMPLING_STEPS = 10_000


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
    for _ in trange(EPOCHS, desc=f"Training boltzmann machine with M={M} hidden neurons", leave=False):
        batch_index = np.random.randint(4, size=BATCH_SIZE)
        bm.train(p_data[batch_index], k=CD_K, lr=LEARNING_RATE)

    # Sample the data distrubution for the model
    frequenzy_table = {i: 0 for i in range(8)}
    for _ in range(1000):
        bm.update() # Remove initial transient
    for _ in range(SAMPLING_STEPS):
        bm.update()
        frequenzy_table[hash_state(bm.visable)] += 1

    # Sample the Kullback-Leibler divergence (only need the XOR data as all other have probability zero -> vanish in D_KL sum)
    p_model = [frequenzy_table[hash_state(p)] / SAMPLING_STEPS for p in p_data]
    D_KL = np.sum([1/4 * (np.log(1/4) - np.log(p_f)) for p_f in p_model])
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


def upperKullbackLeiblerDivergence(N: int, M: np.ndarray) -> np.ndarray:
    """
    Calculate the thoretical upper bound for the Kullback-Leibler divergence
    """
    d_kl = np.zeros_like(M, dtype=float)
    mask = M < 2**(N - 1) - 1
    m = M[mask] + 1.0
    m = N - np.floor(np.log2(m)) - m / (2**np.floor(np.log2(m)))
    d_kl[mask] = np.log(2) * m
    return d_kl


def main():
    M = np.array([1, 2, 4, 8])

    kullback_leibler_samples = np.zeros((AVERAGES, 4))
    for i in trange(AVERAGES, desc="Calculating the average of the Kullback-Leibler divergence"):
        kullback_leibler_samples[i,:] = np.array([train_XOR(m) for m in M])

    kullback_leibler_mean = np.mean(kullback_leibler_samples, axis=0)

    plt.scatter(np.repeat(M, AVERAGES, axis=0), kullback_leibler_samples, marker='x',
     label="Samples from the trained Boltzmann Machine")
    plt.plot(M, kullback_leibler_mean, label=f"Average over {AVERAGES} samples")
    plt.plot(M, upperKullbackLeiblerDivergence(N=3, M=M), label="Theoretical upper bound")
    plt.legend()
    plt.xlabel(r"$M$")
    plt.ylabel(r"$D_\textrm{KL}$")
    plt.show()


if __name__ == '__main__':
    main()
