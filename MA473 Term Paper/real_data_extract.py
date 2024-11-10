import numpy as np 
import pandas as pd
import yfinance as yf 

if __name__ == '__main__':
	stocks = ['ICICIBANK.NS', 'HDFCBANK.NS']
	stock_data = {}

	for stock in stocks:
		ticker = yf.Ticker(stock)
		data = ticker.history(period = "10y")['Close']
		stock_data[stock] = data
		stock_data = pd.DataFrame(stock_data)
		stock_data.to_csv('real_data.csv', index = False)