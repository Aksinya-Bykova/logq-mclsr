import argparse
import os
import pickle
from collections import Counter

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate user/sample counts for user-level LogQ correction.",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to train_mclsr.txt"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save user_counts.pkl"
    )
    parser.add_argument(
        "--num_users",
        type=int,
        default=None,
        help="Number of users from dataset meta. Defaults to max user id in input.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found at {args.input}")

    counts = Counter()
    max_user_id = 0

    with open(args.input, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user_id = int(parts[0])
            counts[user_id] += 1
            max_user_id = max(max_user_id, user_id)

    num_users = args.num_users if args.num_users is not None else max_user_id
    array_size = num_users + 2
    user_counts_array = np.zeros(array_size, dtype=np.float32)

    for user_id, count in counts.items():
        if user_id < array_size:
            user_counts_array[user_id] = count
        else:
            raise ValueError(
                f"user_id {user_id} exceeds array size {array_size}. Check --num_users."
            )

    # =========================================================================
    # BUGFIX & MATH EXPLANATION (LogQ Correction):
    # =========================================================================
    # Previously, this script saved raw interaction counts (e.g., user bought 500 items -> 500.0).
    # In Contrastive Learning, the score (s) is typically a dot product or cosine similarity.
    # The LogQ correction formula dictates: s_corrected = s - lambda * log(p_j).
    #
    # If we subtract raw counts, we get: s_corrected = 0.8 - 500 = -499.2.
    # This completely destroys the logits, killing the Softmax distribution and gradients.
    # We MUST subtract the log probability of a user appearing in a random batch.
    #
    # Correct pipeline:
    # 1. Apply Laplace smoothing (0 -> 1) to avoid log(0) = -inf.
    # 2. Normalize counts to get probabilities: P = count / total_samples.
    # 3. Take the natural logarithm: log(P). (e.g., log(0.01) = -4.6).
    # =========================================================================

    zero_mask = user_counts_array == 0
    user_counts_array[zero_mask] = 1.0  # Laplace smoothing

    # 1. Convert counts to probabilities (P)
    total_samples = np.sum(user_counts_array)
    user_probs_array = user_counts_array / total_samples

    # 2. Apply natural logarithm (log(P))
    log_q_array = np.log(user_probs_array)

    # 3. Save the log(P) array to be used directly in the Loss function
    with open(args.output, "wb") as f:
        pickle.dump(log_q_array, f)

    print(f"Saved log(p) for {len(counts)} users to {args.output}")
