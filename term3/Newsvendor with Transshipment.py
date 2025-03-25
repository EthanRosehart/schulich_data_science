# -*- coding: utf-8 -*-
"""
@author: Adam Diamant (2025)
"""

from gurobipy import Model, GRB
import gurobipy as gb
import pandas as pd
import numpy as np

# Parameters
df = pd.read_csv('distributions.csv')
c = pd.read_csv('cost_matrix.csv').values
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
    x = model.addVars(n, n, scenarios, vtype=GRB.INTEGER, lb=0, name="x")
    o = model.addVars(n, scenarios, vtype=GRB.INTEGER, lb=0, name="o")
    u = model.addVars(n, scenarios, vtype=GRB.INTEGER, lb=0, name="u")
    
    # Objective function
    obj = (1.0/scenarios) * (gb.quicksum(24.44 * o[i, k] for i in range(n) for k in range(scenarios)) 
                   + gb.quicksum(25.55 * u[i, k] for i in range(n) for k in range(scenarios))
                   + gb.quicksum(c[i,j] * x[i,j,k] for i in range(n) for j in range(n) for k in range(scenarios))) 
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    for i in range(n):
        for k in range(scenarios):
            model.addConstr(o[i, k] >= (y[i] + gb.quicksum(x[j,i,k] for j in range(n)) - gb.quicksum(x[i,j,k] for j in range(n))) - D[i][k], name=f"Overage_{i}_{k}")
            model.addConstr(u[i, k] >= D[i][k] - (y[i] + gb.quicksum(x[j,i,k] for j in range(n)) - gb.quicksum(x[i,j,k] for j in range(n))), name=f"Underage_{i}_{k}")
            model.addConstr(gb.quicksum(x[i,j,k] for j in range(n)) <= y[i])
    
    # Optimize the model
    model.optimize()
    
    # The running total
    objectives += model.objVal
    product_ordered += sum(y[i].x for i in range(n))
    
    # Reset the model
    model.reset(0)
    
print("Average Objective: ", objectives/trials) 
print("Average Product Ordered: ", product_ordered/trials)