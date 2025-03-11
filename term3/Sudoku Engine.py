"""
@author: Adam Diamant (2025)
"""

import gurobipy as gb
import matplotlib.pyplot as plt
import random

# How many feasible solutions to generate?
ONE_FEASIBLE = 10

# Create a new model
model = gb.Model("Sudoku Engine")

# Define the size of the Sudoku puzzle where box_size equals N
box_size = 3

# The total number of rows and columns on the board
grid_size = box_size * box_size 
rows = columns = values = range(1, grid_size + 1)

# Decision variables: Binary variable indicating whether a value is xed to a cell
x = model.addVars(rows, columns, values, vtype=gb.GRB.BINARY, name="Cell Value")

# Add Constraints

# Constraint: Each cell can only have one value
model.addConstrs(gb.quicksum(x[i, j, k] for k in values) == 1 for i in rows for j in columns)

# Constraint: Each value can only appear once in each row
model.addConstrs(gb.quicksum(x[i, j, k] for j in values) == 1 for i in rows for k in values)

# Constraint: Each value can only appear once in each column
model.addConstrs(gb.quicksum(x[i, j, k] for i in values) == 1 for j in columns for k in values)

# Constraint: Each value can only appear once in each box
model.addConstrs(gb.quicksum(x[1 + box_size * l + i, 1 + box_size * m + j, k] for i in range(box_size) for j in range(box_size)) == 1
     for l in range(box_size) for m in range(box_size) for k in values)


# Set the condition such that Gurobi continuous searching for feasible solutions
model.setParam('PoolSearchMode', 2)

# Set the number of feasible solutions that Gurobi should look for
model.setParam('PoolSolutions', ONE_FEASIBLE if ONE_FEASIBLE >= 2 else 1)

# Which of the feasible solutions should Gurobi return?
model.setParam('SolutionNumber', random.randrange(ONE_FEASIBLE if ONE_FEASIBLE >= 2 else 1))

# Optimize the model
model.optimize()

# Define the gameboard
cells = [(i, j, k) for i in values for j in values for k in values]

# Print the Sudoku puzzle
if model.status == gb.GRB.OPTIMAL:
    solution = [[0] * grid_size for _ in range(grid_size)]
    for i, j, k in cells:
        if x[i, j, k].xn > 0.5:
            solution[i - 1][j - 1] = k
    
    # Plot the Sudoku solution
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    
    for i in range(grid_size + 1):
        if i % box_size == 0:
            ax.axhline(i, lw=2, color='black')
            ax.axvline(i, lw=2, color='black')
        else:
            ax.axhline(i, lw=0.5, color='black')
            ax.axvline(i, lw=0.5, color='black')
    
    for i in range(grid_size):
        for j in range(grid_size):
            value = solution[i][j]
            if value != 0:
                ax.text(j + 0.5, grid_size - i - 0.5, value, ha='center', va='center', fontsize=10)
    plt.show()
    
else:
    print("No solution found.")

# Number of decision variables in the model
print("Number of Decision Variables: ", model.numVars)

# Number of constraints in the model
print("Number of Constraints: ", model.numConstrs)

# The time it takes to solve the model
print("Model Runtime (s): ", model.Runtime)

# How many feasible solutions were originally generated
print("Solution Count:", model.getAttr('SolCount'))
