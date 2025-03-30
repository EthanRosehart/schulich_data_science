# Part D - Integer Shipping, 10-Year Horizon, WITH Constraints (1–8)

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

#--- 1) Read Data (annual) ----------------------------------------------
capacity_url = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-3/capacity.csv"
demand_url   = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-3/demand.csv"
cr_url       = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-3/costs_revenues.csv"

df_capacity  = pd.read_csv(capacity_url)  
df_demand    = pd.read_csv(demand_url)    
df_cr        = pd.read_csv(cr_url)        

facility_list = df_capacity["Facility"].tolist()   # e.g. [1..17]
region_list   = df_demand["Region"].tolist()       # e.g. [1..31]

capacity_dict   = dict(zip(df_capacity["Facility"], df_capacity["Max_Capacity"]))
fixed_cost_dict = dict(zip(df_cr["Facility"], df_cr["Fixed_Cost"]))

# annual revenue
revenue_dict = {}
for idx, row in df_cr.iterrows():
    f = row["Facility"]
    for r in region_list:
        col_name = f"Revenue_Region_{r}"
        revenue_dict[(f,r)] = row[col_name]

demand_dict = dict(zip(df_demand["Region"], df_demand["Demand"]))

#--- 2) Create Model ----------------------------------------------------
m = gp.Model("AutoParts_FullModel_IntShip")

# Decision vars
y = m.addVars(facility_list, vtype=GRB.BINARY, name="OpenDC")
x = m.addVars(facility_list, region_list, vtype=GRB.INTEGER, lb=0, name="ShipQty")

# Objective: 10-year horizon
m.setObjective(
    gp.quicksum(10.0 * revenue_dict[i,r] * x[i,r] for i in facility_list for r in region_list)
    - gp.quicksum(fixed_cost_dict[i] * y[i] for i in facility_list),
    GRB.MAXIMIZE
)

#--- 3) Core capacity/demand constraints --------------------------------
for i in facility_list:
    m.addConstr(
        gp.quicksum(x[i,r] for r in region_list) <= capacity_dict[i] * y[i],
        name=f"Cap_{i}"
    )

for r in region_list:
    m.addConstr(
        gp.quicksum(x[i,r] for i in facility_list) <= demand_dict[r],
        name=f"Dem_{r}"
    )

#--- 4) Constraints #1–#8 -----------------------------------------------
# 1) At most 3 DCs in {1,2,3,7,11,13,17}
subset1 = [1,2,3,7,11,13,17]
m.addConstr(gp.quicksum(y[i] for i in subset1) <= 3, "Constraint1")

# 2) If DC4 open => DC6, DC8, DC10 open => y4 <= y6, y4 <= y8, y4 <= y10
m.addConstr(y[4] <= y[6],  "Constraint2a")
m.addConstr(y[4] <= y[8],  "Constraint2b")
m.addConstr(y[4] <= y[10], "Constraint2c")

# 3) If DC2 open => at least 2 of {12,14,16} => y12+y14+y16 >= 2*y2
m.addConstr(y[12] + y[14] + y[16] >= 2.0 * y[2], "Constraint3")

# 4) At most 2 DCs among {1,8,9,17}
subset4 = [1,8,9,17]
m.addConstr(gp.quicksum(y[i] for i in subset4) <= 2, "Constraint4")

# 5) If DC1 open => DC5 or DC6 => y1 <= y5 + y6
m.addConstr(y[1] <= (y[5] + y[6]), "Constraint5")

# 6) # DCs in {1..9} <= 1.2 * # DCs in {10..17}
subsetA = range(1,10)  
subsetB = range(10,18)
m.addConstr(
    gp.quicksum(y[i] for i in subsetA) <= 1.2 * gp.quicksum(y[j] for j in subsetB),
    "Constraint6"
)

# 7) sum_{i=1..6} sum_r x[i,r] >= 0.39 * sum_{i=1..17} sum_r x[i,r]
allDCs = range(1,18)
m.addConstr(
    gp.quicksum(x[i,r] for i in range(1,7) for r in region_list)
    >= 0.39 * gp.quicksum(x[i,r] for i in allDCs for r in region_list),
    "Constraint7"
)

# 8) x[i,r] <= 0.45 * demand[r]
for i in facility_list:
    for r in region_list:
        m.addConstr(
            x[i,r] <= 0.45 * demand_dict[r],
            name=f"Constraint8_DC{i}_R{r}"
        )

#--- 5) Solve -----------------------------------------------------------
m.setParam('OutputFlag', 1)
m.setParam("PoolSearchMode", 2)     # 2 = Comprehensive search for solutions
m.setParam("PoolSolutions", 200)    # or however many solutions you'd like to store
m.optimize()

#--- 6) Output ----------------------------------------------------------
if m.status == GRB.OPTIMAL:
    print(f"\nOptimal 10-year Profit (ALL constraints, integer shipping): {m.objVal:,.2f}\n")
    
    for i in facility_list:
        if y[i].X > 0.5:
            shipped = sum(x[i,r].X for r in region_list)
            rev_10 = 10.0 * sum(revenue_dict[(i,r)]*x[i,r].X for r in region_list)
            print(f"DC {i} is OPEN:")
            print(f"  Annual shipments: {shipped:,.0f} units (integer)")
            print(f"  10-year revenue (pre-fixed): ${rev_10:,.2f}")
            print(f"  Fixed cost: ${fixed_cost_dict[i]:,.2f}\n")
else:
    print("No optimal solution found.")

# --- 7) Solution Pool ---------------------------------------------------
# Suppose the model is already optimized with PoolSearchMode and PoolSolutions set
if m.Status == GRB.OPTIMAL or mo.Status == GRB.INT_OPTIMAL:
    bestObj = m.ObjVal
    nSolutions = m.SolCount   # total solutions stored in the solution pool

    within_1pct = []
    
    # Iterate over each solution in the pool
    for k in range(nSolutions):
        m.Params.SolutionNumber = k
        obj_k = m.PoolObjVal    # the objective value of solution k
        # Check if it's within 1% of the best
        if obj_k >= 0.99 * bestObj:
            within_1pct.append(obj_k)
    
    # Now, how many solutions are within 1% of the best?
    count_within_1pct = len(within_1pct)
    # And what's the smallest among them?
    if count_within_1pct > 0:
        smallest_within_1pct = min(within_1pct)
    else:
        smallest_within_1pct = None

    print(f"\nNumber of feasible solutions within 1% of optimal: {count_within_1pct}")
    if smallest_within_1pct is not None:
        print(f"Smallest objective value among those solutions: {smallest_within_1pct:,.2f}")
    else:
        print("No solutions found within 1% of the best objective.")