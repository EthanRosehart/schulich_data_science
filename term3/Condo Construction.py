"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb

# Create the optimization model
model = gb.Model("Condominium Construction")

# Create three classes of five decision variables 
x = model.addVars(7, lb=0, vtype=GRB.INTEGER, name="13-unit floors")
y = model.addVars(7, lb=0, vtype=GRB.INTEGER, name="26-unit floors")
z = model.addVars(7, vtype=GRB.BINARY, name="Site")

# Objective function coefficients
fixed = [3600139.42, 5490819.25, 4881866.51, 3840105.76, 4059055.94, 3394952.48, 5082906.08]
variable_13 = [145044.34, 535503.09, 203022.34, 229679.10, 283722.30, 919371.26, 996211.52]
variable_26 = [1388290.16, 970743.26, 658878.96, 481477.35, 696580.10, 207675.03, 94350.22]

# The objective function
fixed_costs = gb.quicksum(fixed[i]*z[i] for i in range(7))
variable_costs_13 = gb.quicksum(variable_13[i]*x[i] for i in range(7))
variable_costs_26 = gb.quicksum(variable_26[i]*y[i] for i in range(7))
model.setObjective(fixed_costs + variable_costs_13 + variable_costs_26, GRB.MINIMIZE)

# Add the constraints
model.addConstrs((x[i] + y[i] <= 12*z[i] for i in range(7)), "Max Unit Constraint")
model.addConstrs((x[i] + y[i] >= 4*z[i] for i in range(7)), "Min Unit Constraint")
model.addConstrs((x[i] >= 1.0/3*(x[i] + y[i]) for i in range(7)), "Planning Requirement")
model.addConstr(z[3] <= z[6], "Planning Requirement #1")
model.addConstr(z[1] <= z[2] + z[4], "Planning Requirement #2")
model.addConstr(z[0] + z[3] <= 1, "Planning Requirement #3")
model.addConstr(gb.quicksum(z[i] for i in range(7)) == 4, "Number of sites to build")
model.addConstr(gb.quicksum(13*x[i] + 26*y[i] for i in range(7)) >= 916, "Number of units to build")
    
# Optimally solve the problem
model.optimize()

# Print the objective and decision variables
model.printAttr('X')

# The contirbution of each source of costs
print("Fixed Costs: ", fixed_costs.getValue())
print("Variable (13-unit) Costs: ", '%.2f' % variable_costs_13.getValue())
print("Variable (26-unit) Costs: ", '%.2f' % variable_costs_26.getValue())