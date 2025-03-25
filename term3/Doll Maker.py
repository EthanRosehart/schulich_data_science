"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb
import scipy.stats as sp

# Problem parameters
c = 10.0 # cost of procurement
b = 60.0 # lost demand
h = 25.0 # wastage
m = 5.0  # wood per doll

# Create a new optimization model to minimize cost
model = gb.Model("Making Dolls")

# Construct the main decision variables.
x = model.addVar(lb = 0, ub = 1000, vtype = GRB.CONTINUOUS, name="Pounds of wood to procure")

# Construct the cost decision variables for each scenario
y = model.addVars(161, lb=0, vtype=GRB.CONTINUOUS, name="Scenario costs")

#Objective Function = ordering cost + expected future cost for each scenario
#Note that sp.binom.pmf(n,160,0.43) generates the probability of see a demand value of n
#For more information about this function, see: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.binom.html
model.setObjective(c*x + gb.quicksum(sp.binom.pmf(n,160,0.43)*y[n] for n in range(161)), GRB.MINIMIZE)

#Storage constraint 
# model.addConstr(x <= 1000, "Storage")

# Expected future cost constraint
model.addConstrs((y[n] >= b*(n - x * 1/m) for n in range(161)),  "Lost Demand")
model.addConstrs((y[n] >= h*(x * 1/m - n) for n in range(161)), "Wasted Wood")

#Solve our model
model.optimize()

# Number of decision variables in the model
print("Number of Decision Variables: ", model.numVars)

# Number of constraints in the model
print("Number of Constraints: ", model.numConstrs)

# The status of the model
print("Model Status: ", model.status)

# The objective function
print("Objective :", model.objVal)

# Pounds of wood to procure
print("Wood (lbs) to Procure: ", x.x)