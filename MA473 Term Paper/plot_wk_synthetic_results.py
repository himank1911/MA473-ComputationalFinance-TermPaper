import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style("whitegrid")

if __name__ == '__main__':
	for rho in [-1, -0.5, 0, 0.5, 1]:
		results1 = pd.read_csv(f'wk_mjd_results/results1_rho={rho}.csv')
		results2 = pd.read_csv(f'wk_mjd_results/results2_rho={rho}.csv')

		plt.figure(figsize = (12, 6))

		plt.subplot(1, 2, 1)
		plt.scatter(results1['Std Dev'], results1['Mean'], c = results1['Coloring'], cmap = 'winter')
		plt.title(f'Asset 1 Mean-Variance: rho = {rho}')
		plt.xlabel('Std Dev')
		plt.ylabel('Mean')

		plt.subplot(1, 2, 2)
		plt.scatter(results2['Std Dev'], results2['Mean'], c = results2['Coloring'], cmap = 'winter')
		plt.title(f'Asset 2 Mean-Variance: rho = {rho}')
		plt.xlabel('Std Dev')
		plt.ylabel('Mean')

		plt.show()