"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb

# The optimal solutions
optimal_solutions = []

# The parameters
oat_yield = [4.25, 5.1, 3.4]
maize_yield = [3.0, 3.6, 2.4]
soybean_yield = [20.0, 24.0, 16.0]

# Number of options
CROPS = 3
PURCHASED = 2
SOLD = 4

# Selling prices
sell = [220, 260, 55, 26]
purchase = [264, 312]

for k in range(3):

    # Create a new optimization model to maximize profit
    model = gb.Model("Farming Problem")
    
    # Construct the decision variables.
    x = model.addVars(3, lb=0, vtype=GRB.CONTINUOUS, name="Crops")
    y = model.addVars(2, lb=0, vtype=GRB.CONTINUOUS, name="Purchased")
    z = model.addVars(4, lb=0, vtype=GRB.CONTINUOUS, name="Sold")
    
    # Objective Function
    model.setObjective(gb.quicksum(z[i]*sell[i] for i in range(SOLD)) - gb.quicksum(y[i]*purchase[i] for i in range(PURCHASED)), GRB.MAXIMIZE)
    
    # Land capacity constraints 
    model.addConstr(x[0] + x[1] + x[2] <= 500, "Land Capacity")
    
    # Cattle feed constraints (oats)
    model.addConstr(oat_yield[k]*x[0] + y[0] - z[0] >= 200, "Oats")
    
    # Cattle feed constraints (Maize)
    model.addConstr(maize_yield[k]*x[1] + y[1] - z[1] >= 260, "Oats")
    
    # Quota constraints (Soybean)
    model.addConstr(z[2] <= 7000, "Quota")
    model.addConstr(z[2] + z[3] == soybean_yield[k]*x[2], "Soybean")
    
    # Solve our model
    model.optimize()
    
    # Append the objective function value
    optimal_solutions.append(model.objVal)
    
# The Average objective function value
print(optimal_solutions)

# Analyze EVPI 
sp = 369949.33
ws = 0.45 * optimal_solutions[0] + 0.25 * optimal_solutions[1] + 0.3 * optimal_solutions[2]
print("SP Objective Function Value:",  sp)
print("WS Objective Function Value:", ws)
print("EVPI = WS - SP = ", ws - sp)
