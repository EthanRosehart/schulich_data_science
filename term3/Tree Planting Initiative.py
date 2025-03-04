"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb
import numpy as np

# The capacity at each hospital
locations = 36
costs = 0.05 + np.array([1.0/20*(i % 10) for i in range(1,locations+1)])

# Create a new optimization model to minimize the number of clusters
model = gb.Model("Tree Planting Program")

# Define the decision variables and the objective function
x = model.addVars(locations, lb=0, vtype=GRB.INTEGER, name="Trees")
y = model.addVars(locations, vtype=GRB.BINARY, name="Locations")
model.setObjective(gb.quicksum(costs[i]*x[i] for i in range(locations)), GRB.MINIMIZE)

# At least 13 planting locations must be chosen in Ontario.
model.addConstr(gb.quicksum(y[i] for i in range(locations)) >= 13)

# Between 100,000 and 1,000,000 trees can be planted at any location if selected
model.addConstrs(x[i] >= 33111*y[i] for i in range(locations))
model.addConstrs(x[i] <= 668457*y[i] for i in range(locations))

# At most one planting location can be chosen amongst the sites 1, 10, and 20
model.addConstr(y[0] + y[9] + y[19] <= 1)

# No more than 4 planting locations must be chosen amongst the sites 2, 4, 6, 8, 12, 14, and 16
model.addConstr(y[1] + y[3] + y[5] + y[7] + y[11] + y[13] + y[15] <= 4)

# If planting location 30 is chosen then the sites 31 and 32 cannot be chosen.
model.addConstr(y[29] <= 1 - y[30])
model.addConstr(y[29] <= 1 - y[31])

# If planting location 21 is chosen then the sites 22 and 23 must be chosen.
model.addConstr(2*y[20] <= y[21] + y[22])

# The number of planting locations chosen from sites 1-18 must equal the number of planting locations chosen from sites 19-36
model.addConstr(gb.quicksum(y[i] for i in range(18)) == gb.quicksum(y[i] for i in range(18,locations)))

# The total number of 
model.addConstr(gb.quicksum(x[i] for i in range(locations)) == 10000000)

# Solve the integer program
model.optimize()

# Print the optimal solution
model.printAttr('X')