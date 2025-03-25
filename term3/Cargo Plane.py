"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb

# Problem parameters
p = [0.05, 0.15, 0.10, 0.25, 0.30, 0.10, 0.05]
frozen = [0, 0, 4, 8, 12, 16, 20]
refrigerated = [8, 16, 24, 32, 40, 48, 56]
regular = [100, 90, 80, 70, 60, 50, 40]

# Create a new optimization model to minimize costs
model = gb.Model("CargoPlane")

# Problem parameters
scenarios = len(p)
classes = 3

# Construct the decision variables
x = model.addVars(classes, lb=0, vtype=GRB.CONTINUOUS, name="Tons")
y = model.addVars(classes, scenarios, lb=0, vtype=GRB.CONTINUOUS, name="Recourse")
  
#Objective Function
model.setObjective(10*gb.quicksum(p[n]*(4*y[0,n] + 3*y[1,n] + y[2,n]) for n in range(scenarios)), GRB.MAXIMIZE)

# Capacity constraint 
model.addConstr(2*x[0] + 1.5*x[1] + x[2] <= 102, "Capcaity")

# Realization constraints
for n in range(scenarios):
    model.addConstr(y[0,n] <= frozen[n], "Frozen Demand")
    model.addConstr(y[0,n] <= x[0], "Frozen Capacity")
    model.addConstr(y[1,n] <= refrigerated[n], "Fridge Demand")
    model.addConstr(y[1,n] <= x[1], "Frozen Capacity")
    model.addConstr(y[2,n] <= regular[n], "Regular Demand")
    model.addConstr(y[2,n] <= x[2], "Regular Capacity")
      
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

# Capacity to assign
print("(Frozen, Refrigerated, Regular): ", (x[0].x, x[1].x, x[2].x))