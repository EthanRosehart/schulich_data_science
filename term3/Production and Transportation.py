"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb

# Create the optimization model
model = gb.Model("Production and Transportation")

# Create three classes of five decision variables 
x = model.addVars(8, 8, lb=0, vtype=GRB.INTEGER, name="Shipping")
y = model.addVars(8, vtype=GRB.BINARY, name="Production Facility")
z = model.addVars(8, vtype=GRB.BINARY, name="Customer Served")

# Objective function coefficients
revenues = [75740, 44370, 46320, 87780, 43850, 21000, 74850, 83980]
quantity = [1430, 870, 770, 1140, 700, 830, 1230, 1070]
distance = [[0, 983, 1815, 1991, 3036, 1539, 213, 2664], [983, 0, 1205, 1050, 2112, 1390, 840, 1729], 
            [1815, 1205, 0, 801, 1425, 1332, 1604, 1027], [1991, 1050, 801, 0, 1174, 2065, 1780, 836],
            [3036, 2112, 1425, 1174, 0, 2757, 2825, 398], [1539, 1390, 1332, 2065, 2757, 0, 1258, 2359],
            [213, 840, 1604, 1780, 2825, 1258, 0, 2442], [2664, 1729, 1027, 836, 398, 2359, 2442, 0]]


# The objective function
revenue = gb.quicksum(revenues[i]*z[i] for i in range(8))
fixed_costs = 60000*gb.quicksum(y[i] for i in range(8))
production_costs = 10.25*gb.quicksum(x[i,j] for i in range(8) for j in range(8))
transportation_costs = 0.02*gb.quicksum(distance[i][j] * x[i,j] for i in range(8) for j in range(8))
model.setObjective(revenue - fixed_costs - production_costs - transportation_costs, GRB.MAXIMIZE)

# Add the constraints
for i in range(8):
    model.addConstr(gb.quicksum(x[i,j] for j in range(8)) <= 2500*y[i], "Resource Constraint %i" %i)
    
for j in range(8):
    model.addConstr(gb.quicksum(x[i,j] for i in range(8)) == quantity[j]*z[j], "Demand Constraint %i" %j)
      
# Optimally solve the problem
model.optimize()

# Print the objective and decision variables
model.printAttr('X')

# The contirbution of each source of costs
print("Revenue: ", revenue.getValue())
print("Fixed Costs: ", fixed_costs.getValue())
print("Production Costs: ", production_costs.getValue())
print("Transportation Costs: ", transportation_costs.getValue())

# Number of decision variables in the model
print("Number of Decision Variables: ", model.numVars)

# Number of constraints in the model
print("Number of Constraints: ", model.numConstrs)

# The status of the model
print("Model Status: ", model.status)
