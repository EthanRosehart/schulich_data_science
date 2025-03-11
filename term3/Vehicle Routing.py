"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb
import pandas as pd
import ast
from itertools import permutations

# ----------------------------------------------------------------------------------------
# Read the travel time data from the csv file
hubs = pd.read_csv("hubs.csv", converters={'key': ast.literal_eval})
time = dict(zip(hubs['key'], hubs['value']))

# Parameters including the number of vehicles, customers to visit, and total shift time.
vehicles = 4
customers_including_depot = 16
vans = range(vehicles)
locations = range(customers_including_depot) 
shift_time = 240

# Should we include subtour elmination constriants? 
# Notice what happens when we toggle this variable on and off.
INCLUDE_SUBTOUR = True
# ----------------------------------------------------------------------------------------
# Lazy Constraints

# Callback - use lazy constraints to eliminate sub-tours
# There are an exponential number of these constraints!
# Thus, we don't add them all to the model. Instead, we
# use a callback function to find violated subtour constraints 
# and add them to the model as lazy constraints.
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        
        # Make a list of all pairs of nodes that are selected in the solution for all vans
        vals = model.cbGetSolution(model._x)
        selected = gb.tuplelist((i,j) for i, j, k in model._x.keys() if vals[i, j, k] > 0.5)
        
        # Find the shortest cycle (from hub to locations and back to the hub)
        tour = subtour(selected)
        
        # If the length of these tours does not 
        if len(tour) < customers_including_depot: 
            for k in vans:
                model.cbLazy(gb.quicksum(model._x[i, j, k] for i, j in permutations(tour, 2)) <= len(tour)-1)


# Given a tuplelist of edges, find the shortest subtour not containing depot (0)
def subtour(edges):
    # Track the locations that have not been visted
    unvisited = list(range(1, customers_including_depot))
    
    # A cycle should start and end at the depot
    cycle = range(customers_including_depot+1)
    
    # Iterate until there are no unvisited customers
    while unvisited:
        
        # Track the cycle associated with the unvisited nodes
        thiscycle = []
        neighbors = unvisited
        
        # Look to see who the next visited customer is in the the current tour
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            if current != 0:
                unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*') if j == 0 or j in unvisited]
        
        # If 0 is not contained in this tour, we must create a constraint that 
        # prevents this solution from occuring which is then added to the problem.
        if 0 not in thiscycle and len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle

# ----------------------------------------------------------------------------------------
# Create a new model
model = gb.Model("Delivery Routing")

# Decision variables: Binary variables

# Equals 1 if van k visits location i and then goes to location j
x = model.addVars(locations, locations, vans, vtype=GRB.BINARY, name = "Routes")

# Equals 1 if location i is visited by van k
y = model.addVars(locations, vans, vtype=GRB.BINARY, name = "Locations")

# Equals 1 if van k is used
z = model.addVars(vans, vtype=GRB.BINARY, name = 'Vans')

# Define the objective function as minimizing the number of vans used
model.setObjective(gb.quicksum(z[k] for k in vans), gb.GRB.MINIMIZE)

# Add Constraints

# Constraint: If a van visits at least one location, it is used
model.addConstrs(y[i,k] <= z[k] for i in locations for k in vans if i > 0)

# Constraint: Travel time + service time (15 minutes per delivery) for each van must not exceed the shift time (minutes)
model.addConstrs(gb.quicksum(time[i,j]*x[i,j,k] for i in locations for j in locations if i != j) + 15*gb.quicksum(y[i,k] for i in locations if i > 0) <= shift_time for k in vans) 

# Constraint: Each customer must be visited
model.addConstrs(gb.quicksum(y[i,k] for k in vans) == 1 for i in locations if i > 0)

# Constraint: Each van must visit the depot if it is in use
model.addConstrs(y[0,k] == z[k] for k in vans)

# Constraint: If a van k arrives at location j, it has come from some location i
model.addConstrs(gb.quicksum(x[i,j,k] for i in locations) == y[j,k] for j in locations for k in vans)

# Constraint: If a van k leaves location j, it must be going to location i
model.addConstrs(gb.quicksum(x[i,j,k] for j in locations) == y[i,k] for i in locations for k in vans)

# Constraint: The van cannot travel between the same location
model.addConstrs(x[i,i,k] == 0 for i in locations for k in vans)

# Optimize the model using lazy constraints
if INCLUDE_SUBTOUR:
    model._x = x
    model.Params.LazyConstraints = 1
    
    # During the optimization, the "callback" function will be called periodically.
    # You can see how many times it was called by looking at "User-callback calls" in the output.
    model.optimize(subtourelim)
else:
    model.optimize()

# Retrieve the optimal solution
if model.status == gb.GRB.OPTIMAL:
    # Print optimal routes for each van
    for k in vans:
        route = gb.tuplelist((i,j) for i,j in time.keys() if x[i,j,k].X > 0.5)
        if route:
            i = 0
            print(f"Route for van {k}: {i}", end='')
            while True:
                i = route.select(i, '*')[0][1]
                print(f" -> {i}", end='')
                if i == 0:
                    break
        else:
            print(f"Route for van {k}:", end='')
        print("")
else:
    print("No solution found.")

# Number of decision variables in the model
print("Number of Decision Variables: ", model.numVars)

# Number of constraints in the model
print("Number of Constraints: ", model.numConstrs)

# Value of the objective
print("Number of vans: ", model.objval)
