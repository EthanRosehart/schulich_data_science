# Part C - Integer Shipping, 10-Year Horizon, No Constraints (1–8)

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

#--- 1) Read Data ------------------------------------------------------
capacity_url = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-3/capacity.csv"
demand_url   = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-3/demand.csv"
cr_url       = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-3/costs_revenues.csv"

df_capacity  = pd.read_csv(capacity_url)  # Facility, Max_Capacity (annual)
df_demand    = pd.read_csv(demand_url)    # Region, Demand (annual)
df_cr        = pd.read_csv(cr_url)        # Facility, Fixed_Cost, Revenue_Region_1..31

facility_list = df_capacity["Facility"].tolist()   # e.g. [1..17]
region_list   = df_demand["Region"].tolist()       # e.g. [1..31]

capacity_dict   = dict(zip(df_capacity["Facility"], df_capacity["Max_Capacity"]))
fixed_cost_dict = dict(zip(df_cr["Facility"], df_cr["Fixed_Cost"]))

# Build a dictionary of annual revenue for each (DC,Region)
revenue_dict = {}
for idx, row in df_cr.iterrows():
    f = row["Facility"]
    for r in region_list:
        col_name = f"Revenue_Region_{r}"
        revenue_dict[(f, r)] = row[col_name]

demand_dict = dict(zip(df_demand["Region"], df_demand["Demand"]))

#--- 2) Create Model (No constraints #1–#8) ----------------------------
m = gp.Model("Benchmark_10year_IntShip")

# Decision variables:
#  y[i]: 1 if DC i is open, else 0
#  x[i,r]: integer number of units shipped per year from DC i to region r
y = m.addVars(facility_list, vtype=GRB.BINARY, name="OpenDC")
x = m.addVars(facility_list, region_list, vtype=GRB.INTEGER, lb=0, name="ShipQty")

# Objective: 10 * sum of annual revenue - fixed cost
m.setObjective(
    gp.quicksum(10.0 * revenue_dict[(i,r)] * x[i,r] for i in facility_list for r in region_list)
    - gp.quicksum(fixed_cost_dict[i] * y[i] for i in facility_list),
    GRB.MAXIMIZE
)

# Capacity constraints (annual)
for i in facility_list:
    m.addConstr(
        gp.quicksum(x[i,r] for r in region_list) <= capacity_dict[i] * y[i],
        name=f"Cap_{i}"
    )

# Demand constraints (annual)
for r in region_list:
    m.addConstr(
        gp.quicksum(x[i,r] for i in facility_list) <= demand_dict[r],
        name=f"Dem_{r}"
    )

# Solve
m.setParam('OutputFlag', 1)
m.optimize()

#--- 3) Results ---------------------------------------------------------
if m.status == GRB.OPTIMAL:
    print(f"\nOptimal 10-year Profit (integer shipping, no constraints 1–8): {m.objVal:,.2f}\n")
    
    # Print which DCs are open, shipping, etc.
    for i in facility_list:
        if y[i].X > 0.5:
            total_annual_ship = sum(x[i,r].X for r in region_list)
            total_10yr_rev = 10.0 * sum(revenue_dict[(i,r)]*x[i,r].X for r in region_list)
            print(f"DC {i} is OPEN:")
            print(f"  Annual shipments: {total_annual_ship:,.0f} units (integer)")
            print(f"  10-year revenue (before fixed): ${total_10yr_rev:,.2f}")
            print(f"  Fixed cost: ${fixed_cost_dict[i]:,.2f}\n")
else:
    print("No optimal solution found.")
