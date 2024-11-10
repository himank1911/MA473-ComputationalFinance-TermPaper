import numpy as np
import pandas as pd
from wkmeans import wasserstein_distance, wasserstein_barycentre, wk_means

def empirical_cdf(segment, x):
    N = len(segment)
    return np.sum([x >= r for r in segment]) / N

if __name__ == '__main__':
	
	for rho in [-1, -0.5, 0, 0.5, 1]:	
		data = pd.read_csv(f'mjd_data/mjd_data_rho={rho}.csv')
		
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
			means1.append(segments[i]['Asset 1'].mean())
			stds1.append(segments[i]['Asset 1'].std())

			means2.append(segments[i]['Asset 2'].mean())
			stds2.append(segments[i]['Asset 2'].std())

		K = 2
		clusters1 = wk_means(M, K, segments, ['Asset 1'])
		clusters2 = wk_means(M, K, segments, ['Asset 2'])
		
		coloring1 = []
		coloring2 = []

		for i in range(M):
		    W1 = [wasserstein_distance(segments[i][['Asset 1']], segments[clusters1[j]][['Asset 1']]) for j in range(K)]
		    coloring1.append(np.argmin(W1))

		    W2 = [wasserstein_distance(segments[i][['Asset 2']], segments[clusters2[j]][['Asset 2']]) for j in range(K)]
		    coloring2.append(np.argmin(W2))

		results1 = pd.DataFrame({
			'Mean': means1,
			'Std Dev': stds1,
			'Coloring': coloring1
		})
		results1.to_csv(f'wk_mjd_results/results1_rho={rho}.csv', index = False)

		results2 = pd.DataFrame({
			'Mean': means2,
			'Std Dev': stds2,
			'Coloring': coloring2
		})
		results2.to_csv(f'wk_mjd_results/results2_rho={rho}.csv', index = False)










