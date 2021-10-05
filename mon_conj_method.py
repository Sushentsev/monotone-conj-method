import argparse
from typing import Tuple

import numpy as np
from scipy.stats import rankdata


def get_ranks(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Calculates y ranks.
    :param xs: array of x values
    :param ys: array of y values
    :return: ranks
    """
    sorted_idx = np.argsort(xs)
    xs, ys = xs[sorted_idx], ys[sorted_idx]
    return rankdata(-ys)


def check_mon_conj(xs: np.ndarray, ys: np.ndarray) -> Tuple[int, int, float]:
    """
    Calculates indicators of monotonic conjugation.
    :param xs: array of x values
    :param ys: array of y values
    :return: tuple of R_1 - R_2, std, conjugacy rate
    """
    N = len(xs)
    if N < 9:
        raise ValueError("Number of points must be not less than 9")

    p = round(N / 3)
    ranks = get_ranks(xs, ys)
    R1, R2 = sum(ranks[:p]), sum(ranks[-p:])
    std = (N + 0.5) * np.sqrt(p / 6)
    conj_rate = (R1 - R2) / (p * (N - p))
    return int(R1 - R2), round(std), round(conj_rate, 2)


def read_data(in_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads data from input file.
    Each line of the input file consists of a pair (x, y).
    :param in_path: input file path
    :return: arrays of x an y values
    """
    xs, ys = [], []
    with open(in_path, "r") as file:
        for line in file:
            x, y = tuple(map(float, line.strip().split()))
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)


def save_result(result: Tuple[int, int, float], out_path: str):
    """
    Saves results separated by space to output file.
    :param result: method result
    :param out_path: output file path
    :return:
    """
    with open(out_path, "w") as file:
        file.write(" ".join(map(str, result)) + "\n")


def main(in_path: str, out_path: str):
    xs, ys = read_data(in_path)
    results = check_mon_conj(xs, ys)
    save_result(results, out_path)


if __name__ == "__main__":
    __parser = argparse.ArgumentParser(description="Monotonic conjugation method")
    __parser.add_argument("--in_path", type=str, default="./data/in.txt", help="Path to input file")
    __parser.add_argument("--out_path", type=str, default="./data/out.txt", help="Path to output file")
    __args = __parser.parse_args()

    main(__args.in_path, __args.out_path)
