import numpy as np
import pandas as pd
from wkmeans import wasserstein_distance, wasserstein_barycentre, wk_means

def empirical_cdf(segment, x):
    N = len(segment)
    return np.sum([x >= r for r in segment]) / N

if __name__ == '__main__':	
	data = pd.read_csv('real_data.csv')
	stocks = data.columns
	
	log_returns = np.log(data / data.shift(1))[1:]
	log_returns.reset_index(inplace = True)
	log_returns.drop('index', axis = 1, inplace = True)

	h1 = 15
	h2 = 8
	N = len(log_returns)

	segments = []
	for i in range(h1, N, h1 - h2):
		segment = log_returns[i - h1: i]
		segment.reset_index(inplace = True)
		segment = segment.drop('index', axis = 1)
		segments.append(segment)

	M = len(segments)

	means1 = []
	stds1 = []

	means2 = []
	stds2 = []

	for i in range(M):
		means1.append(segments[i][stocks[0]].mean())
		stds1.append(segments[i][stocks[0]].std())

		means2.append(segments[i][stocks[1]].mean())
		stds2.append(segments[i][stocks[1]].std())

	K = 2

	clusters1 = wk_means(M, K, segments, [stocks[0]])
	clusters2 = wk_means(M, K, segments, [stocks[1]])
	
	coloring1 = []
	coloring2 = []

	for i in range(M):
	    W1 = [wasserstein_distance(segments[i][[stocks[0]]], segments[clusters1[j]][[stocks[1]]]) for j in range(K)]
	    coloring1.append(np.argmin(W1))

	    W2 = [wasserstein_distance(segments[i][[stocks[0]]], segments[clusters2[j]][[stocks[1]]]) for j in range(K)]
	    coloring2.append(np.argmin(W2))

	results1 = pd.DataFrame({
		'Mean': means1,
		'Std Dev': stds1,
		'Coloring': coloring1
	})
	results1.to_csv('wk_real_results/results1.csv', index = False)

	results2 = pd.DataFrame({
		'Mean': means2,
		'Std Dev': stds2,
		'Coloring': coloring2
	})
	results2.to_csv('wk_real_results/results2.csv', index = False)


	transformed_segments = [segments[i].copy(deep = True) for i in range(M)]
	
	for i in range(M):
		for asset in stocks:
			segment = transformed_segments[i][asset]
			transformed_segments[i][asset] = [empirical_cdf(segment, r) for r in segment]

	corr_clusters = wk_means(M, K, transformed_segments, stocks)

	correlations = []
	coloring = []

	for i in range(M):
		correlations.append(transformed_segments[i].corr()[stocks[0]][stocks[1]])
		W = [wasserstein_distance(transformed_segments[i], transformed_segments[corr_clusters[j]]) for j in range(K)]
		coloring.append(np.argmin(W))

	results_corr = pd.DataFrame({
		'Correlation': correlations,
		'Coloring': coloring
	})
	results_corr.to_csv('wk_real_results/results_corr.csv')










