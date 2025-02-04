# -*- coding: utf-8 -*-
"""
@author: Adam Diamant (2025)
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# Load the training data
train_file = "Student Ridge Regression - Training Data.xlsx"
train_data = pd.read_excel(train_file)

# Define the features and outcome columns
y_train = train_data['Grade'].values                 # outcomes
X_train = train_data.drop(columns=['Grade']).values  # feature matrix
N, J = X_train.shape                                 # number of training instances (N) and features (J)

# Budget constraint for regularization
t = 0.01

# Create a new Gurobi model
model = gp.Model("Ridge Regression")

# Add decision variables: alpha (scalar), beta (vector of length J), u_i (vector of length N), v_i (vector of length N)
alpha = model.addVar(vtype=GRB.CONTINUOUS, name="alpha")
beta = model.addVars(J, vtype=GRB.CONTINUOUS, name="beta")

# Set the objective function as the squared error between predicted and actual
# model.addConstrs(predicted[i] ==  for i in range(N)) 
objective = (1.0 / N) * gp.quicksum( (y_train[i] - alpha - gp.quicksum(beta[j] * X_train[i, j] for j in range(J)))**2 for i in range(N))
model.setObjective(objective, GRB.MINIMIZE)

# Regularization constraint
budget = model.addConstr(gp.quicksum(beta[j]*beta[j] for j in range(J)) <= t)

# Optimize the model
model.optimize()

# Extract the optimal values of alpha and beta
alpha_opt = alpha.X
beta_opt = np.array([beta[j].X for j in range(J)])
mean_grade = np.mean(y_train)

print(f"Optimal alpha: {alpha_opt}")
print(f"Optimal beta: {beta_opt}")

lhs = sum(beta[j].X ** 2 for j in range(J))
print(f'Budget constraint slack: {t - lhs}')

# Load the testing data
test_file = "Student Ridge Regression - Testing Data.xlsx"
test_data = pd.read_excel(test_file)

# Define the features and outcome columns for testing data
y_test = test_data['Grade'].values  # outcomes
X_test = test_data.drop(columns=['Grade']).values  # feature matrix

# Predict outcomes for the testing data
y_pred = alpha_opt + X_test.dot(beta_opt)

# Calculate Mean Absolute Deviation (MAD) and Mean Squared Error (MSE)
mse = np.mean((y_test - y_pred) ** 2)

# Mean Squared Error (MSE) on benchmark
mse_bench = np.mean((y_test - mean_grade) ** 2)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Squared Error (MSE) Benchmark: {mse_bench}")
