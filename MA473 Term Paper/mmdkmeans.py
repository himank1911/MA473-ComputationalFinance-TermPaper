import numpy as np
import pandas as pd
import math
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Kernel Function
def rbf_kernel(x, y, sigma=1.0):
    gamma = 1 / (2 * sigma ** 2)
    return np.exp(-gamma * cdist(x, y, 'sqeuclidean'))

# Maximum Mean Discrepancy
def maximum_mean_discrepancy(X, Y, sigma=1.0):
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)
    n = X.shape[0]
    m = Y.shape[0]
    mmd_value = (np.sum(K_XX) / (n * n)) + (np.sum(K_YY) / (m * m)) - (2 * np.sum(K_XY) / (n * m))
    
    return np.sqrt(mmd_value)

#Barycentre
def barycentre(clustering, segments, stocks):
    n = len(clustering)
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            w = maximum_mean_discrepancy(segments[clustering[i]][stocks], segments[clustering[j]][stocks])
            W[i, j] = w
            W[j, i] = w

    W_rowsums = [np.sum(W[i]) for i in range(n)]
    q = np.argmin(W_rowsums)
    return clustering[q]


# MMDK-Means Algorithm
def mmdk_means(M, K, segments, stocks):
    NUM_ITERATIONS = 10
    clusters_old = np.random.choice(np.arange(0, M), size = K)

    for it in range(NUM_ITERATIONS):
        clusterings = {}
        for i in range(K):
            clusterings[i] = []

        for i in range(M):
            W = [maximum_mean_discrepancy(segments[i][stocks], segments[clusters_old[j]][stocks]) for j in range(K)]
            c = np.argmin(W)
            clusterings[c].append(i)

        clusters_new = [barycentre(clustering, segments, stocks) for clustering in clusterings.values()]
        print(stocks, ": Iteration", it, ": ", "Clusters =", clusters_new)

        if np.array_equal(clusters_new, clusters_old):
            clusters_old = clusters_new
            break

        clusters_old = clusters_new

    return clusters_old
