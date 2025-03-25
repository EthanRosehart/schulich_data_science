"""
@author: Adam Diamant (2025)
"""

import random
import math

# Calculate the price of the house for a given set of parameters
def geometric_brownian_motion(S,v,r,T):
    return S * math.exp((r - 0.5 * v**2) * T + v * math.sqrt(T) * random.gauss(0,1.0))

# Calculate the payoff of the European-style put option at maturity
def put_option_payoff(S_T,K):
    return max(K-S_T,0.0)
  
S = 1400000  # the current price of the house + renovation input
v = 0.05     # the annualised standard deviation of the assets returns
r = 0.027    # the risk free interest rate
T = 1.0      # the time to maturity 
K = 1500000  # the strike price

# The number of trials to perform
TRIALS = 1000000
totalPrice = 0
totalPriceSquared = 0

# The Monte Carlo simulation
for trial in range(0,TRIALS):
    S_T = geometric_brownian_motion(S,v,r,T)
    optionPrice = math.exp(-r * T)*put_option_payoff(S_T, K)
    totalPrice += optionPrice                                    # Add the option price to the summation
    totalPriceSquared += optionPrice*optionPrice                 # Add the option price squared to the summation
    
# Calculate the average time over all trials and print the result 
averagePrice = 1.0*totalPrice/TRIALS
print("The average price of the option is $%2.2f." % averagePrice)    

# Calculate the standard error over all trials and print the result 
variance = 1.0/(TRIALS-1)*totalPriceSquared - 1.0*TRIALS/(TRIALS-1)*averagePrice*averagePrice
standardDeviation = math.sqrt(variance)              
standardError = math.sqrt(1.0*variance/TRIALS)
print("The standard deviation is $%2.2f." % standardDeviation)
print("The standard error is $%2.2f." % standardError)
print("The 90%% confidence interval is (%2.2f, %2.2f)." % (averagePrice -  1.645*standardError , averagePrice + 1.645*standardError))
print("The 95%% confidence interval is (%2.2f, %2.2f)." % (averagePrice -  1.96*standardError , averagePrice + 1.96*standardError))
print("The 99%% confidence interval is (%2.2f, %2.2f)." % (averagePrice -  2.575*standardError , averagePrice + 2.575*standardError))
