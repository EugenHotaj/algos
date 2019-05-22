"""An simple implementation of k-Means Clustering."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster


def kmeans(data, num_clusters):
  """Performs k-means clustering on the given `data`.

  Args:
    data: The data to cluster, where each row represents one data point.
    num_clusters: The number of clusters to cluster the data into.

  Returns:
    An integer array `clusters` of the same size and in the same order as
    `data`. Two points in `data` i, j are in the same cluster if clusters[i] ==
    clusters[j].
  """

  # Select random initial cluster centers.
  centers = np.split(data, num_clusters)
  centers = np.array(
      [center[np.random.randint(len(center))] for center in centers])

  clusters = np.zeros((len(data),))
  while True:
    for i, point in enumerate(data):
      distances = np.linalg.norm(centers - point, axis=1)
      clusters[i] = np.argmin(distances)

    delta = 0
    for i, c in enumerate(centers):
      members = np.where(clusters == i)[0]
      if not members.size:
        continue
      new_center = np.mean(data[members], axis=0)
      delta += np.linalg.norm(new_center - c)
      centers[i] = new_center
    if delta < .0001:
      break

  return clusters


if __name__ == '__main__':
  cluster_0 = np.random.normal(loc=[0, 0], scale=1.0, size=(25, 2))
  cluster_1 = np.random.normal(loc=[-3, -1], scale=1.0, size=(25, 2))
  cluster_2 = np.random.normal(loc=[2, -2], scale=1.0, size=(25, 2))

  data = np.concatenate((cluster_0, cluster_1, cluster_2))
  np.random.shuffle(data)
  clusters = kmeans(data, 3)

  plt.scatter(data[:, 0], data[:, 1], 5, clusters)
  plt.show()
