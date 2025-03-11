# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:45:32 2024

@author: Adam Standard
"""
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Load the data from the CSV file
data_file = 'cutting_stock_data.csv'
data = pd.read_csv(data_file)

# Extract the order lengths and demands from the CSV file
order_lens = data['Order_Length'].values
demands = data['Demand'].values
num_orders = len(order_lens)
total_demand = np.sum(demands)

# Set the roll length to the maximum order length (to ensure feasibility)
roll_len = max(order_lens)

# Create Gurobi model
model = gp.Model("Cutting_Stock_Optimization")

# Set ranges (rolls and orders)
rolls = range(total_demand)
orders = range(num_orders)

# Decision variables
# Binary variable for whether a roll is used (1 if used, 0 otherwise)
x = model.addVars(rolls, vtype=GRB.BINARY, obj=np.ones(total_demand), name="X")

# Integer variable for how much of each order is cut from each roll
y = model.addVars(orders, rolls, vtype=GRB.INTEGER, obj=np.zeros((num_orders, total_demand)), name="Y")

# Direction of optimization: Minimization
model.modelSense = GRB.MINIMIZE

# Demand satisfaction constraint: Each order's demand must be satisfied exactly
model.addConstrs((y.sum(i, '*') == demands[i] for i in orders), "Demand")

# Roll length constraint: The sum of order lengths assigned to a roll cannot exceed the roll length
for j in rolls:
    model.addConstr(sum(y[i, j] * order_lens[i] for i in orders) <= roll_len, f"Length[{j}]")

# Linking constraint: Ensure that orders are assigned to a roll only if the roll is used
for i in orders:
    for j in rolls:
        model.addConstr(y[i, j] <= x[j] * demands[i])

# Optimize the model
model.optimize()

# Extract the solution
if model.status == GRB.OPTIMAL:
    used_rolls = [j for j in rolls if x[j].x > 0.5]
    print(f'Maximum Rolls: {total_demand}')
    print(f'Number of Orders: {num_orders}')
    print(f'Standard Roll Size: {roll_len}')
    print(f'Objective: {model.objVal}')
    