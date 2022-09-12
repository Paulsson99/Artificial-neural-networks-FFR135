"""
We want to estimate the one step error probability

The mathematic explaining this can be found in the README document in this folder
"""

import numpy as np
from argparse import ArgumentParser


def cross_talk_term(N: int, p: int, M: int, zero_diagonal: bool) -> np.ndarray:
    """
    Calculate the cross talk term by taking samples from a binomial distrubution
    """
    K = (N - 1) * (p - 1)
    binomial_dist = np.random.binomial(K, 0.5, M)
    C = 2 * binomial_dist - K
    if not zero_diagonal:
        C += p - 1
    return -C / N


def main():
    N = 120
    ps = [12,24,48,70,100,120]
    M = 10_000_000
    zero_diagonal = True
    threshold = 1
    if zero_diagonal:
        threshold -= 1/N
    for p in ps:
        cross_talk = cross_talk_term(N, p, M, zero_diagonal)
        m = np.count_nonzero(cross_talk > threshold)
        print(f"One step error probability for N={N} and p={p}: {m/M:.4f}")


if __name__ == '__main__':
    main()
