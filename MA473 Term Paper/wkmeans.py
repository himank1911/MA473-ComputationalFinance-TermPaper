import numpy as np
import pandas as pd
import math
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# p-Wasserstein Distance
def wasserstein_distance(segment1, segment2):
    N = len(segment1)
    M = len(segment1.columns)
    C = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            X = np.array(segment1.iloc[i]).reshape((1, M))
            Y = np.array(segment2.iloc[j]).reshape((1, M))
            C[i, j] = cdist(X, Y) ** 2

    row_ind, col_ind = linear_sum_assignment(C)
    return (C[row_ind, col_ind].sum()) ** 0.5


# Barycentre
def wasserstein_barycentre(clustering, segments, stocks):
    n = len(clustering)
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            w = wasserstein_distance(segments[clustering[i]][stocks], segments[clustering[j]][stocks])
            W[i, j] = w
            W[j, i] = w

    W_rowsums = [np.sum(W[i]) for i in range(n)]
    q = np.argmin(W_rowsums)
    return clustering[q]


# WK-Means Algorithm
def wk_means(M, K, segments, stocks):
    NUM_ITERATIONS = 10
    clusters_old = np.random.choice(np.arange(0, M), size = K)

    for it in range(NUM_ITERATIONS):
        clusterings = {}
        for i in range(K):
            clusterings[i] = []

        for i in range(M):
            W = [wasserstein_distance(segments[i][stocks], segments[clusters_old[j]][stocks]) for j in range(K)]
            c = np.argmin(W)
            clusterings[c].append(i)

        clusters_new = [wasserstein_barycentre(clustering, segments, stocks) for clustering in clusterings.values()]
        print(stocks, ": Iteration", it, ": ", "Clusters =", clusters_new)

        if np.array_equal(clusters_new, clusters_old):
            clusters_old = clusters_new
            break

        clusters_old = clusters_new

    return clusters_old




