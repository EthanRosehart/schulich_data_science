# -*- coding: utf-8 -*-
"""
@author: Adam Diamant (2025)
"""
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# Load instance data from CSV
def load_instance_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    
    # Extract the order lengths and demands from the CSV file
    order_lens = data['Order_Length'].values
    roll_len = max(order_lens)
    demands = data['Demand'].values
    num_orders = len(order_lens)
    return roll_len, order_lens, demands, num_orders

# Main function to run column generation algorithm iterating between master/subproblem
def column_generation_and_solve(roll_len, order_lens, demands):    
    # Generate initial patterns (one per order, filling as much as possible)
    patterns = []
    for i in range(len(order_lens)):
        pattern = list(np.zeros(len(order_lens)).astype(int))
        pattern[i] = int(roll_len / order_lens[i])
        patterns.append(pattern)

    # Column generation loop
    while True:
        
        # These details change because in each iteration, we have one more pattern
        n_pattern = len(patterns)
        pattern_range = range(n_pattern)
        order_range = range(len(order_lens))
        patterns_np = np.array(patterns, dtype=int)

        # Master problem setup
        master_problem = gp.Model("master_problem")
        
        # We have to define it like this because there are lambda functions 
        # in Python (https://realpython.com/python-lambda/). There is one variable per pattern.
        # We can also c
        lambda_ = master_problem.addVars(pattern_range, vtype=GRB.CONTINUOUS, lb = 0, name="lambda")

        # Demand satisfaction constraints
        for i in order_range:
            master_problem.addConstr(sum(patterns_np[p, i] * lambda_[p] for p in pattern_range) == demands[i],
                                     "Demand[%d]" % i)

        # Objective: minimize the number of rolls used
        master_problem.setObjective(gp.quicksum(lambda_[p] for p in pattern_range), GRB.MINIMIZE)
        master_problem.optimize()        

        # Retrieve dual variables from the demand constraints
        duals = np.array([constraint.pi for constraint in master_problem.getConstrs()])

        # Solve subproblem (pricing problem)
        subproblem = gp.Model("subproblem")
        a = subproblem.addVars(order_range, vtype=GRB.INTEGER, lb=0, name="a")
       
        # The feasible length constraint
        subproblem.addConstr(sum(order_lens[i] * a[i] for i in order_range) <= roll_len)
        
        # Find the most negative reduced cost
        subproblem.setObjective(gp.quicksum(duals[i]*a[i] for i in order_range), GRB.MAXIMIZE)
        subproblem.optimize()

        # Check if new pattern has a reduced cost > -1 (i.e., stop condition) or z > 1
        if subproblem.objVal < 1 + 0.001:
            break

        # Add new pattern to the master problem
        new_pattern = [int(a[i].x) for i in order_range]
        patterns.append(new_pattern)

    # Solve the final integer master problem
    return patterns

# Solve the final master problem as an integer programming problem
def solve_integer_master_problem(patterns, order_lens, demands, roll_len):
    n_pattern = len(patterns)
    pattern_range = range(n_pattern)
    order_range = range(len(order_lens))
    patterns_np = np.array(patterns, dtype=int)

    # Integer Master Problem setup
    int_master_problem = gp.Model("integer_master_problem")
    lambda_int = int_master_problem.addVars(pattern_range, vtype=GRB.INTEGER, lb = 0, name="lambda_int")

    # Demand satisfaction constraints
    for i in order_range:
        int_master_problem.addConstr(sum(patterns_np[p, i] * lambda_int[p] for p in pattern_range) >= demands[i],
                                     "Demand[%d]" % i)

    # Objective: minimize the number of rolls used (integer version)   
    int_master_problem.setObjective(gp.quicksum(lambda_int[p] for p in pattern_range), GRB.MINIMIZE)
    int_master_problem.optimize()        

    # Print the solution
    print("\nFinal Integer Solution (Roll Usage):")
    for p in pattern_range:
        if lambda_int[p].x > 0:
            print(f"Pattern {p}: Used {int(lambda_int[p].x)} times")

    return int_master_problem.objVal


# Script to read data and solve the problem
if __name__ == "__main__":
    # Load instance from CSV (CSV format: roll_len, order_len, demand columns)
    csv_file = "cutting_stock_data.csv"
    roll_len, order_lens, demands, num_orders = load_instance_from_csv(csv_file)
    
    # Run column generation and solve the final integer problem
    patterns = column_generation_and_solve(roll_len, order_lens, demands)
    final_patterns = solve_integer_master_problem(patterns, order_lens, demands, roll_len)
    print(f'Number of Orders: {num_orders}')
    print(f'Standard Roll Size: {roll_len}')
    print(f'Objective: {final_patterns }')
