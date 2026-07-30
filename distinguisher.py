"""
ML distinguisher for round-reduced SM4.

Methodology (standard "real-vs-random" neural/ML distinguisher setup, as used
in differential-neural cryptanalysis, e.g. Gohr's work on Speck):

For a fixed plaintext input difference `delta`:
  - "Real" sample: pick a random key K and random plaintext P0. Let
    P1 = P0 XOR delta. Encrypt both under K for R rounds -> (C0, C1).
    Label = 1.
  - "Random" sample: pick a random key K and random plaintext P0, encrypt
    it to get C0. Replace C1 with an independent random 16-byte string
    (i.e. NOT the encryption of P0 XOR delta). Label = 0.

Feature vector = bits of (C0 XOR C1) concatenated with bits of C0 (giving the
model both the ciphertext difference and absolute ciphertext values, which
is known to help).

We train a classifier per round-count R and report test accuracy. For a
secure cipher, accuracy should collapse to ~0.5 (coin flip) after only a
handful of rounds -- that collapse is itself the interesting security result.
"""

import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from sm4 import sm4_encrypt

RNG = np.random.default_rng(1337)


def bytes_to_bits(arr_bytes: np.ndarray) -> np.ndarray:
    """arr_bytes: (N, 16) uint8 -> (N, 128) float32 bit array."""
    return np.unpackbits(arr_bytes, axis=1).astype(np.float32)


def make_dataset(n_samples: int, rounds: int, delta: bytes, seed: int = 0):
    rng = np.random.default_rng(seed)
    half = n_samples // 2

    c0_real = np.zeros((half, 16), dtype=np.uint8)
    c1_real = np.zeros((half, 16), dtype=np.uint8)
    c0_rand = np.zeros((n_samples - half, 16), dtype=np.uint8)
    c1_rand = np.zeros((n_samples - half, 16), dtype=np.uint8)

    delta_arr = np.frombuffer(delta, dtype=np.uint8)

    for i in range(half):
        key = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
        p0 = rng.integers(0, 256, size=16, dtype=np.uint8)
        p1 = np.bitwise_xor(p0, delta_arr)
        c0 = sm4_encrypt(key, p0.tobytes(), rounds=rounds)
        c1 = sm4_encrypt(key, p1.tobytes(), rounds=rounds)
        c0_real[i] = np.frombuffer(c0, dtype=np.uint8)
        c1_real[i] = np.frombuffer(c1, dtype=np.uint8)

    n_rand = n_samples - half
    for i in range(n_rand):
        key = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
        p0 = rng.integers(0, 256, size=16, dtype=np.uint8)
        c0 = sm4_encrypt(key, p0.tobytes(), rounds=rounds)
        c1_random_bytes = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
        c0_rand[i] = np.frombuffer(c0, dtype=np.uint8)
        c1_rand[i] = np.frombuffer(c1_random_bytes, dtype=np.uint8)

    c0 = np.vstack([c0_real, c0_rand])
    c1 = np.vstack([c1_real, c1_rand])
    y = np.concatenate([np.ones(half, dtype=np.int64), np.zeros(n_rand, dtype=np.int64)])

    diff = np.bitwise_xor(c0, c1)
    x_bits = np.concatenate([bytes_to_bits(diff), bytes_to_bits(c0)], axis=1)  # (N, 256)

    # shuffle
    perm = rng.permutation(n_samples)
    return x_bits[perm], y[perm]


def train_and_eval(rounds: int, n_train: int = 20000, n_test: int = 5000,
                    delta: bytes = bytes([0] * 15 + [1])):
    x_train, y_train = make_dataset(n_train, rounds, delta, seed=1000 + rounds)
    x_test, y_test = make_dataset(n_test, rounds, delta, seed=5000 + rounds)

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        alpha=1e-4,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    return acc, auc


def main():
    delta = bytes([0] * 15 + [1])  # single-bit input difference in the last byte
    round_counts = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]

    results = []
    print(f"{'rounds':>6} | {'accuracy':>8} | {'ROC-AUC':>8}")
    print("-" * 30)
    for r in round_counts:
        acc, auc = train_and_eval(r)
        results.append((r, acc, auc))
        print(f"{r:>6} | {acc:>8.4f} | {auc:>8.4f}")

    # Save results to CSV for the report / plot
    with open("results.csv", "w") as f:
        f.write("rounds,accuracy,auc\n")
        for r, acc, auc in results:
            f.write(f"{r},{acc:.6f},{auc:.6f}\n")

    return results


if __name__ == "__main__":
    main()
