"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb

# Create a new optimization model to maximize profit
model = gb.Model("Farming Problem")

# Number of options
CROPS = 3
PURCHASED = 2
SOLD = 4

# Selling prices
sell = [220, 260, 55, 26]
purchase = [264, 312]

# Construct the decision variables.
x = model.addVars(3, lb=0, vtype=GRB.CONTINUOUS, name="Crops")
y = model.addVars(2, 3, lb=0, vtype=GRB.CONTINUOUS, name="Purchased")
z = model.addVars(4, 3, lb=0, vtype=GRB.CONTINUOUS, name="Sold")

# Objective Function
average = gb.quicksum(z[i,0]*sell[i] for i in range(SOLD)) - gb.quicksum(y[i,0]*purchase[i] for i in range(PURCHASED))
optimistic = gb.quicksum(z[i,1]*sell[i] for i in range(SOLD)) - gb.quicksum(y[i,1]*purchase[i] for i in range(PURCHASED))
pessimistic = gb.quicksum(z[i,2]*sell[i] for i in range(SOLD)) - gb.quicksum(y[i,2]*purchase[i] for i in range(PURCHASED))
model.setObjective(0.25*optimistic + 0.45*pessimistic + 0.30*average, GRB.MAXIMIZE)

# Land capacity constraints 
model.addConstr(x[0] + x[1] + x[2] <= 500, "Land Capacity")

# Cattle feed constraints (Oats)
model.addConstr(4.25*x[0] + y[0,0] - z[0,0] >= 200, "Oats Scenario 1")
model.addConstr(5.1*x[0] + y[0,1] - z[0,1] >= 200, "Oats Scenario 2")
model.addConstr(3.4*x[0] + y[0,2] - z[0,2] >= 200, "Oats Scenario 3")

# Cattle feed constraints (Maize)
model.addConstr(3.00*x[1] + y[1,0] - z[1,0] >= 260, "Maize Scenario 1")
model.addConstr(3.60*x[1] + y[1,1] - z[1,1] >= 260, "Maize Scenario 2")
model.addConstr(2.40*x[1] + y[1,2] - z[1,2] >= 260, "Maize Scenario 3")

# Quota constraints (Soybean)
model.addConstr(z[2,0] <= 7000, "Quota Scenario 1")
model.addConstr(z[2,1] <= 7000, "Quota Scenario 2")
model.addConstr(z[2,2] <= 7000 , "Quota Scenario 3")
model.addConstr(z[2,0] + z[3,0] == 20*x[2], "Soybean Scenario 1")
model.addConstr(z[2,1] + z[3,1] == 24*x[2], "Soybean Scenario 2")
model.addConstr(z[2,2] + z[3,2] == 16*x[2], "Soybean Scenario 3")

# Solve our model
model.optimize()

# Number of decision variables in the model
print("Number of Decision Variables: ", model.numVars)

# Number of constraints in the model
print("Number of Constraints: ", model.numConstrs)

# The status of the model
print("Model Status: ", model.status)

# The objective function
print("Objective :", model.objVal)

# Crops to plant
print("Acres Planted (Oats, Maize, Soybean): ", ('%.2f' % x[0].x, '%.2f' % x[1].x, '%.2f' % x[2].x))