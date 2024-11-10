import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style("whitegrid")

if __name__ == '__main__':
	results1 = pd.read_csv(f'mmdk_real_results/results1.csv')
	results2 = pd.read_csv(f'mmdk_real_results/results2.csv')
	results_corr = pd.read_csv(f'mmdk_real_results/results_corr.csv')

	stocks = ['ICICI Bank', 'HDFC Bank']

	plt.figure(figsize = (14, 6))

	plt.subplot(1, 3, 1)
	plt.scatter(results1['Std Dev'], results1['Mean'], c = results1['Coloring'], cmap = 'winter')
	plt.title(f'{stocks[0]} Mean-Variance')
	plt.xlabel('Std Dev')
	plt.ylabel('Mean')

	plt.subplot(1, 3, 2)
	plt.scatter(results2['Std Dev'], results2['Mean'], c = results2['Coloring'], cmap = 'winter')
	plt.title(f'{stocks[1]} Mean-Variance')
	plt.xlabel('Std Dev')
	plt.ylabel('Mean')

	plt.subplot(1, 3, 3)
	plt.scatter(np.arange(0, len(results_corr)), results_corr['Correlation'], c = results_corr['Coloring'], cmap = 'winter')
	plt.title(f'{stocks[0]} - {stocks[1]} Correlation')
	plt.xlabel('Index')
	plt.ylabel('Correlations')

	plt.show()