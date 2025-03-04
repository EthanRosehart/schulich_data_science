"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb

# Create the optimization model
model = gb.Model("In-House vs. Outsourcing Manufacturing")

# Objective function coefficients
fixed = [1000, 1200, 1900, 1500, 1500]
produce = [0.4, 2.9, 3.15, 0.3, 0.55]
purchase = [0.65, 3.45, 3.7, 0.5, 0.7]

# The demand coefficients
D = [12000, 7000, 5000, 7000, 5000]

# Create three classes of five decision variables 
x = model.addVars(5, lb=0, vtype=GRB.CONTINUOUS, name="Manufacture")
y = model.addVars(5, vtype=GRB.BINARY, name="Toggle")
z = model.addVars(5, lb=0, vtype=GRB.CONTINUOUS, name="Outsource")

# The objective function
fixed_costs = gb.quicksum(fixed[i]*y[i] for i in range(5))
production_costs = gb.quicksum(produce[i]*x[i] for i in range(5))
purchase_costs = gb.quicksum(purchase[i]*z[i] for i in range(5))
model.setObjective(fixed_costs + production_costs + purchase_costs, GRB.MINIMIZE)

# Add the constraints
for i in range(5):
    model.addConstr(x[i] + z[i] == D[i], "Demand Constraint %i" %i)
    model.addConstr(x[i] <= D[i]*y[i], "Big-M Constraint %i" %i)
model.addConstr(0.9*x[0] + 2.2*x[1] + 3*x[2] + 0.8*x[3] + x[4] <= 30000, "Capacity Constraint")
    
# Optimally solve the problem
model.optimize()

# Print the objective and decision variables
model.printAttr('X')

# The contirbution of each source of costs
print("Fixed Costs: ", fixed_costs.getValue())
print("Production Costs: ", production_costs.getValue())
print("Purchase Costs: ", purchase_costs.getValue())
