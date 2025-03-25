"""
@author: Adam Diamant (2025)
"""

import random
import math
import numpy

# Should we calculate and report the SE?
CALCULATE_SE = True

def simulate_monthly_sales(selling_weeks_left, target, trials=2000):
    
    # A metric to track in how many trials the target is achieved
    target_achieved = 0
        
    for trial in range(trials):
        
        # The total number of sales ($) during the selling days
        sales_cumulative = 0 
        
        # The number of selling weeks
        for week in range(selling_weeks_left):
            
            # The number of sales per week
            number_of_sales = numpy.random.poisson(5)
            
            # The size of each sale 
            for sales in range(number_of_sales):            
                sales_cumulative += random.lognormvariate(6, 3)       # Generates a lognormal random variate           
        
        # After the remaining selling horizon, is the target achieved?
        if (sales_cumulative >= target):
            target_achieved += 1
            
    # Return the probability that the target is achieved over all trials
    return target_achieved/trials

trials = 100000
prob = simulate_monthly_sales(9, 187000, trials)
print("The probability is %2.3f." % prob)    


if CALCULATE_SE:
    # Calculate the standard error over all trials and print the result 
    standardDeviation = math.sqrt(prob * (1-prob))
    standardError = standardDeviation/math.sqrt(trials)
    print("The standard deviation is %2.3f." % standardDeviation)
    print("The standard error is %2.3f." % standardError)
    print("The 90%% confidence interval is (%2.3f, %2.3f)." % (prob -  1.645*standardError , prob + 1.645*standardError))
    print("The 95%% confidence interval is (%2.3f, %2.3f)." % (prob -  1.96*standardError , prob + 1.96*standardError))
    print("The 99%% confidence interval is (%2.3f, %2.3f)." % (prob -  2.575*standardError , prob + 2.575*standardError))