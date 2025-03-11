"""
@author: Adam Diamant (2025)
"""

import matplotlib.pyplot as plt
from gurobipy import GRB
import gurobipy as gb
import pandas as pd
import yfinance as yf
import numpy as np
from math import sqrt

# Compute minimum risk portfolio or efficient frontier?
FRONTIER = True

# Should we download the data?
READ_FILES = True

if READ_FILES:

    # Read the closing prices and ticker symbols
    df = pd.read_csv("stock_closing_prices.csv")
    symbols = pd.read_csv("symbols.csv")
    stocks = symbols["Symbol"].values.tolist()

    # Matrix of daily closing prices for each stock in the S&P 500
    closes = np.transpose(np.array(df)) 

else:
    
    # Read the ticker symbols
    symbols = pd.read_csv("symbols.csv")
    
    # Read the ticker symbols of the S&P 500 from the file    
    stocks = symbols["Symbol"].values.tolist()
    
    # Download two years worth of data for each stock from Yahoo Finance
    data = yf.download(stocks, period='2y')

    # We only need the closing prices
    df = data.Close
    
    # Write this data to a file
    # df.to_csv("stock_closing_prices.csv", index=False)
    
    # Matrix of daily closing prices for each stock in the S&P 500
    closes = np.transpose(np.array(data.Close)) 


# The absolute change in daily closing prices for each stock on the S&P 500
absdiff = np.diff(closes)                   

# Compute the daily return for each stock on the S&P 500 by dividing the 
# absolute difference in closes prices by the starting share price. Note
# that this normalizes the return so it doesn't depend on share prices.
reldiff = np.divide(absdiff, closes[:,:-1]) 

# The mean return for each stoch on the S&P 500
mu = np.mean(reldiff, axis=1)

# The standard deviation of returns (diagonal of the covariance matrix)
std = np.std(reldiff, axis=1)               

# The convariance matrix associated with the returns 
sigma = np.cov(reldiff)                     

# Find the nan values for mu and std
nan_indices_mu = np.isnan(mu)
nan_indices_std = np.isnan(std)
nan_indices_combined = np.logical_or(nan_indices_mu, nan_indices_std) 

# Remove the nan values for mu, std, and sigma
mu = mu[~nan_indices_combined]
std = std[~nan_indices_combined]
sigma = sigma[~nan_indices_combined][:, ~nan_indices_combined]

# Create an empty optimization model
model = gb.Model('Portfolio Optimization')

# Add decision variables for the non-nan stocks
stock_index = range(len(mu))
x = model.addVars(stock_index, lb=0, vtype=gb.GRB.CONTINUOUS, name="Fraction")

# Objective is to minimize risk.  This is modeled using the
# covariance matrix, which measures the historical correlation between stocks
portfolio_risk = gb.quicksum(x[i]*x[j]*sigma[i,j] for i in stock_index for j in stock_index)
model.setObjective(portfolio_risk, GRB.MINIMIZE)

# The proportion constraints ensure we invest our entire portfolio
model.addConstr(gb.quicksum(x[i] for i in stock_index) == 1)

# Optimize model to find the minimum risk portfolio
model.optimize()

# Create an array of proportions which represent the optimal solution
x_flat = np.array([x[i].x for i in stock_index])

# Comptue the minimum risk of the portfolio as well as the expected return (daily)
minrisk_volatility = sqrt(model.objval)
minrisk_return = mu @ x_flat

# Convert the average daily values into a yearly value (251 working days).
# Then, convert these yearly values into a percentage.
number_of_days = len(closes[0])/2       # This equals 251
minrisk_return_out = minrisk_return*number_of_days*100
minrisk_volatility_out = minrisk_volatility*sqrt(number_of_days)*100

# Print the composition of the portfolio the minimizes the risk.
primary_investments = [i for i in range(len(mu)) if x_flat[i] > 0.01]
filtered_stocks = [stock for i, stock in enumerate(stocks) if not nan_indices_combined[i]]
for i in primary_investments:
    print(filtered_stocks[i], x_flat[i])
    
# Print out the return and volatility of the portfolio
print("Expected Yearly Return (%): ", minrisk_return_out)
print("Expected Yearly Volatility (%): ", minrisk_volatility_out)
    
# Did you want to compute the efficient frontier?
if FRONTIER:

    # Create an expression representing the expected return for the portfolio; add this as a constraint
    target = model.addConstr(gb.quicksum(mu[i]*x[i] for i in stock_index) >= minrisk_return, 'target')
    
    # Solve for efficient frontier by varying the mean return
    frontier = np.empty((2,0))
    for r in np.linspace(mu.min(), mu.max(), 25):
        target.rhs = r
        model.optimize()
        frontier = np.append(frontier, [[sqrt(model.objval)],[r]], axis=1)
    
    # Plot the efficient frontier
    fig, ax = plt.subplots(figsize=(10,8))
    
    # Plot volatility versus expected return for individual stocks
    ax.scatter(x=std, y=mu, color='Blue', label='Individual Stocks')
    for i, stock in enumerate(filtered_stocks):
        ax.annotate(stock, (std[i], mu[i]))
    
    # Plot volatility versus expected return for minimum risk portfolio
    ax.scatter(x=minrisk_volatility, y=minrisk_return, color='DarkGreen')
    ax.annotate('Minimum\nRisk\nPortfolio', (minrisk_volatility, minrisk_return), horizontalalignment='right')
    
    # Plot efficient frontier
    ax.plot(frontier[0], frontier[1], label='Efficient Frontier', color='DarkGreen')
    
    # Format and display the final plot
    ax.axis([frontier[0].min()*0.7, frontier[0].max()*1.3, mu.min()*1.2, mu.max()*1.2])
    ax.set_xlabel('Volatility (standard deviation)')
    ax.set_ylabel('Expected Return')
    ax.legend()
    ax.grid()
    plt.show()
    