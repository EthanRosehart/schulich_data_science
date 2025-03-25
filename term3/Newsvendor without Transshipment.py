# -*- coding: utf-8 -*-
"""
@author: Adam Diamant (2025)
"""

from gurobipy import Model, GRB
import gurobipy as gb
import pandas as pd
import numpy as np

# Parameters
df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/distributions.csv')
n = len(df)

# The sum of all objectives and decision variables
objectives = 0
product_ordered = 0

# The number of trials to perform
trials = 50
    
# The number of scenarios per trial
scenarios = 100
    
# Create a new model
model = Model("Multilocation Newsvendor Problem")
model.setParam('OutputFlag', 0)

for trial in range(trials):  
    
    # Create the random demand for this trials
    D = np.zeros((n, scenarios))

    # Generate samples for each row
    for index, row in df.iterrows():        
        sample = np.random.binomial(n=row['n'], p=row['p'], size=scenarios)
        D[index, :] = sample

    # Decision variables
    y = model.addVars(n, vtype=GRB.INTEGER, lb=0, name="y")
    o = model.addVars(n, scenarios, vtype=GRB.INTEGER, lb=0, name="o")
    u = model.addVars(n, scenarios, vtype=GRB.INTEGER, lb=0, name="u")
    
    # Objective function
    obj = (1.0/scenarios) * (gb.quicksum(24.44 * o[i, k] for i in range(n) for k in range(scenarios)) + gb.quicksum(25.55 * u[i, k] for i in range(n) for k in range(scenarios)))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    for i in range(n):
        for k in range(scenarios):
            model.addConstr(o[i, k] >= y[i] - D[i][k], name=f"Overage_{i}_{k}")
            model.addConstr(u[i, k] >= D[i][k] - y[i], name=f"Underage_{i}_{k}")
    
    # Optimize the model
    model.optimize()
    
    if model.status == GRB.Status.OPTIMAL:
        
        # The running total
        objectives += model.objVal
        product_ordered += sum(y[i].x for i in range(n))
        
    # Reset the model
    model.reset(0)
    
print("Average Objective: ", objectives/trials) 
print("Average Product Ordered: ", product_ordered/trials)