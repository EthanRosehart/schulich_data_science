"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb

# Create a new optimization model
model = gb.Model("Coal Mining")

# The number of scenarios
demand_scenarios = 5
mining_scenarios = 6

# The values in each scenario
demand = [81,92,103,114,125]  # Daily energy demand (demand rate)
mine = [5,10,15,20,25,30]   # The amount of coal mined (mining rate)

# Declare the recourse variable
x = model.addVar(lb=0, ub=24, vtype=GRB.CONTINUOUS, name="Mining Hours")
y = model.addVars(demand_scenarios, mining_scenarios, lb=0, vtype=GRB.CONTINUOUS, name="Recourse")

#First-stage (Cost for total number of hours for first-stage)
first_stage = 8800*x

#Second-stage (cost of purchasing energy)
second_stage = gb.quicksum(y[i,j] for i in range(demand_scenarios) for j in range(mining_scenarios))

#Objective Function to minimize expected costs
model.setObjective(first_stage + 2000.0/30.0 * second_stage, GRB.MINIMIZE)

#Second-stage constraints 
model.addConstrs((mine[j]*x + y[i,j] >= demand[i] for i in range(demand_scenarios) for j in range(mining_scenarios)), "Demand Satisfaction")

#Solve our model
model.optimize()   

# The contirbution of each source of costs
print("Fixed Costs: ", first_stage.getValue())
print("Energy Costs: ", second_stage.getValue())

# Value of the first-stage variable
print("Mining Hours per Day: ", x.x)