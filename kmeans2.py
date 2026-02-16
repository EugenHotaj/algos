import argparse
import math 
import random

import matplotlib.pyplot as plt
from sklearn import datasets


def distance(left: list[float], right: list[float]) -> float:
    dist = 0
    for l, r in zip(left, right, strict=True):
        dist += math.pow(l - r, 2)
    return math.sqrt(dist)


def k_means(Xs: list[list[float]], n_centers: int, eps=1e-6) -> list[int]:
    n_features = len(Xs[0])
    old_centers = [random.choice(Xs) for _ in range(n_centers)]
    cluster_idxs = [0 for _ in range(len(Xs))]
    while True:
        # One round of k-means.
        new_centers = [[0.0 for _ in range(n_features)] for _ in range(n_centers)]
        n_per_center = [0 for _ in range(n_centers)]
        for i, x in enumerate(Xs):
            # Figure out which cluster each point belongs to.
            min_distance, min_idx = 1e9, -1
            for j, c in enumerate(old_centers):
                d = distance(x, c)
                if d < min_distance:
                    min_distance, min_idx = d, j
            assert j != -1
            
            # Sum up contributions from each point to the new cluster.
            for j in range(n_features):
                new_centers[min_idx][j] += x[j]
            n_per_center[min_idx] += 1

            # Update cluster ids.
            cluster_idxs[i] = min_idx

        # Normalize sums to get new centers.
        for i, n in enumerate(n_per_center):
            if n == 0:
                continue
            for j in range(n_features):
                new_centers[i][j] /= n

        # Check if we are done.
        dist = 0
        for old_c, new_c in zip(old_centers, new_centers):
            dist += distance(old_c, new_c)
        old_centers = new_centers
        if dist <= eps:
            break

    return old_centers, cluster_idxs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-centers", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="blobs")
    args = parser.parse_args()

    n_centers = args.n_centers
    assert 1 <= n_centers <= 10, "centers must be in [1, 10]"

    if args.dataset == "moons":
        Xs, _ = datasets.make_moons(n_samples=300, noise=0.08)
    elif args.dataset.startswith("blobs"):
        n_blobs = int(args.dataset.split("::", 1)[1]) if "::" in args.dataset else 3
        assert 1 <= n_blobs <= 10, "blobs must be in [1, 10]"
        Xs, _ = datasets.make_blobs(n_samples=300, centers=n_blobs, n_features=2)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    Xs = Xs.tolist()
    centers, idxs = k_means(Xs, n_centers)

    point_colors = ["blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan", "black"]
    plt.scatter(
        [x[0] for x in Xs],
        [x[1] for x in Xs],
        c=[point_colors[i % len(point_colors)] for i in idxs],
        s=10,
    )
    plt.scatter([c[0] for c in centers], [c[1] for c in centers], c="red", marker="x", s=120)
    plt.show()
