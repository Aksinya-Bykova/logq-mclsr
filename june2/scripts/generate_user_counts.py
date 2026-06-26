import argparse
import os
import pickle
from collections import Counter

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate user/sample counts for user-level LogQ correction.",
    )
    parser.add_argument("--input", type=str, required=True, help="Path to train_mclsr.txt")
    parser.add_argument("--output", type=str, required=True, help="Path to save user_counts.pkl")
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

    with open(args.input, 'r') as f:
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

    zero_mask = user_counts_array == 0
    user_counts_array[zero_mask] = 1.0

    with open(args.output, 'wb') as f:
        pickle.dump(user_counts_array, f)

    print(f"Saved user counts for {len(counts)} users to {args.output}")


if __name__ == "__main__":
    main()
