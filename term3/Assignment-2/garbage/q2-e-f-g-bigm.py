#!/usr/bin/env python3
import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd

# ============================================================
# 1) DATA LOADING
# ============================================================
# Load hotels.csv. Expected columns: Room_ID, Floor, Square_Feet, Cleaning_Time_Hours
df = pd.read_csv("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/hotels.csv")
# We assume each row is a room. We’ll use the dataframe index as room identifier.
rooms = list(df.index)
# Create a dictionary for cleaning times (in hours)
cleaning_time = {j: df.loc[j, "Cleaning_Time_Hours"] for j in rooms}

# ============================================================
# 2) PARAMETERS & MODEL SETUP
# ============================================================
num_attendants = 8
attendants = range(num_attendants)

regular_wage = 25.0   # $25 per hour
base_hours   = 8.0    # Base hours paid
max_hours    = 10.0   # Maximum hours allowed per attendant

# Big-M for overtime indicators (since overtime can be at most 2 hours)
BIG = 2.0
epsilon = 1e-3  # small tolerance

# Create the Gurobi model.
m = gp.Model("HotelStaffing_BigM")

# ------------------------------------------------------------
# 2a) Decision Variables
# ------------------------------------------------------------
# x[i,j]: 1 if attendant i cleans room j
x = m.addVars(attendants, rooms, vtype=GRB.BINARY, name="x")

# y[i]: 1 if attendant i is employed (assigned at least one room)
y = m.addVars(attendants, vtype=GRB.BINARY, name="y")

# H[i]: total cleaning hours for attendant i (can be fractional)
H = m.addVars(attendants, lb=0.0, name="H")

# To pay full overtime hours, we introduce two binary indicators:
# u1[i] = 1 if H[i] > 8 (i.e. at least one overtime hour is required)
# u2[i] = 1 if H[i] > 9 (i.e. a second overtime hour is needed)
u1 = m.addVars(attendants, vtype=GRB.BINARY, name="u1")
u2 = m.addVars(attendants, vtype=GRB.BINARY, name="u2")

# OT[i]: overtime hours for attendant i (integer: 0, 1, or 2)
OT = m.addVars(attendants, vtype=GRB.INTEGER, lb=0, ub=2, name="OT")

# ------------------------------------------------------------
# 2b) Constraints
# ------------------------------------------------------------
# (1) Each room must be cleaned by exactly one attendant.
for j in rooms:
    m.addConstr(quicksum(x[i, j] for i in attendants) == 1, name=f"Room_{j}_covered")

# (2) Link room assignments to employment.
for i in attendants:
    for j in rooms:
        m.addConstr(x[i, j] <= y[i], name=f"Link_x_y_{i}_{j}")

# (3) Total cleaning hours for each attendant.
for i in attendants:
    m.addConstr(H[i] == quicksum(cleaning_time[j] * x[i, j] for j in rooms),
                name=f"TotalHours_{i}")
    # If not employed, H[i] must be 0; if employed, H[i] is capped at max_hours.
    m.addConstr(H[i] <= max_hours * y[i], name=f"MaxHours_{i}")

# (4) Overtime logic:
# We want to charge full overtime hours. The idea is:
#    u1[i] = 1 if H[i] > 8; u2[i] = 1 if H[i] > 9.
# Then force OT[i] = u1[i] + u2[i].
for i in attendants:
    # If u1[i] == 0 then H[i] <= 8.
    m.addGenConstrIndicator(u1[i], 0, H[i] <= base_hours, name=f"u1_zero_{i}")
    # If u1[i] == 1 then H[i] >= 8 + epsilon.
    m.addGenConstrIndicator(u1[i], 1, H[i] >= base_hours + epsilon, name=f"u1_one_{i}")
    # If u2[i] == 0 then H[i] <= 9.
    m.addGenConstrIndicator(u2[i], 0, H[i] <= 9.0, name=f"u2_zero_{i}")
    # If u2[i] == 1 then H[i] >= 9 + epsilon.
    m.addGenConstrIndicator(u2[i], 1, H[i] >= 9.0 + epsilon, name=f"u2_one_{i}")
    # Force OT[i] = u1[i] + u2[i]
    m.addConstr(OT[i] == u1[i] + u2[i], name=f"OT_def_{i}")

# ------------------------------------------------------------
# 3) OBJECTIVE FUNCTION
# ------------------------------------------------------------
# Base cost: each employed attendant is paid for 8 hours at $25 (i.e. $200)
# Overtime cost: each overtime hour is paid at 1.5x $25 = $37.50.
# Total cost = sum_{i} [200*y[i] + 37.5*OT[i]]
total_cost = quicksum(200.0 * y[i] + 37.5 * OT[i] for i in attendants)
m.setObjective(total_cost, GRB.MINIMIZE)

m.update()
m.optimize()

# ============================================================
# 4) REPORT INTEGER SOLUTION RESULTS
# ============================================================
print("\n=== Integer Model Results ===")
if m.status == GRB.OPTIMAL:
    print("Total Optimal Cost = $%.2f" % m.objVal)
    for i in attendants:
        assigned_rooms = [j for j in rooms if x[i, j].X > 0.5]
        if assigned_rooms:
            print(f"\nAttendant {i}:")
            print(f"  Rooms Covered: {assigned_rooms}")
            print(f"  Total Cleaning Hours: {H[i].X:.2f}")
            print(f"  Overtime Hours (integer): {OT[i].X}")
            print(f"  Cost for Attendant: ${200.0*y[i].X + 37.5*OT[i].X:.2f}")
        else:
            print(f"\nAttendant {i}: Not employed")
else:
    print("No optimal integer solution found.")

# ============================================================
# 5) SOLVE THE RELAXED MODEL AND REPORT RESULTS
# ============================================================
relaxed_model = m.relax()  # relax all integrality constraints
relaxed_model.optimize()

print("\n=== Relaxed Model Results ===")
if relaxed_model.status == GRB.OPTIMAL:
    print("Total Optimal Cost (Relaxation) = $%.2f" % relaxed_model.objVal)
    for i in attendants:
        # Retrieve variable values from the relaxed model.
        xi = [j for j in rooms if relaxed_model.getVarByName(f"x[{i},{j}]").X > 0.5]
        Hi = relaxed_model.getVarByName(f"H[{i}]").X
        OTi = relaxed_model.getVarByName(f"OT[{i}]").X
        yi = relaxed_model.getVarByName(f"y[{i}]").X
        print(f"\nAttendant {i}:")
        print(f"  Rooms Covered: {xi}")
        print(f"  Total Cleaning Hours: {Hi:.2f}")
        print(f"  Overtime Hours (integer variable, may be fractional in relaxation): {OTi:.2f}")
        print(f"  Employed (y): {yi:.2f}")
else:
    print("No optimal relaxed solution found.")


# -------------------------
# MANUAL LINEAR RELAXATION (Part g)
# -------------------------
# Create a copy of the original model
M_relax_manual = m.copy()

# Loop over all variables in the copy and change binary/integer types to continuous with bounds [0,1]
for var in M_relax_manual.getVars():
    if var.VType in ['B', 'I']:
        var.VType = GRB.CONTINUOUS
        var.lb = 0
        var.ub = 1

M_relax_manual.update()
M_relax_manual.optimize()

# Build helper dictionaries to retrieve relaxed variable values by their original indices.
e_relaxed = {}
T_relaxed = {}
y1_relaxed = {}
y2_relaxed = {}

for var in M_relax_manual.getVars():
    name = var.VarName
    if name.startswith("e["):
        # Expected format: "e[<i>]"
        idx = int(name.split('[')[1].split(']')[0])
        e_relaxed[idx] = var
    elif name.startswith("T["):
        idx = int(name.split('[')[1].split(']')[0])
        T_relaxed[idx] = var
    elif name.startswith("y1["):
        idx = int(name.split('[')[1].split(']')[0])
        y1_relaxed[idx] = var
    elif name.startswith("y2["):
        idx = int(name.split('[')[1].split(']')[0])
        y2_relaxed[idx] = var

print("\n=== Manual Linear Relaxation Results ===")
if M_relax_manual.status == GRB.OPTIMAL:
    print("Manual Linear Relaxation Optimal Total Cost = $%.2f" % M_relax_manual.objVal)
    for i in attendants:
        # For x, we can still use getVarByName (they usually retain their names)
        assigned_rooms = [j for j in rooms if M_relax_manual.getVarByName(f"x[{i},{j}]").X > 0.5]
        e_val  = e_relaxed[i].X if i in e_relaxed else 0.0
        T_val  = T_relaxed[i].X if i in T_relaxed else 0.0
        y1_val = y1_relaxed[i].X if i in y1_relaxed else 0.0
        y2_val = y2_relaxed[i].X if i in y2_relaxed else 0.0
        OT_val = y1_val + 2 * y2_val
        print(f"\nAttendant {i}:")
        print(f"  Rooms Covered: {assigned_rooms}")
        print(f"  Employed (e): {e_val:.2f}")
        print(f"  Total Cleaning Hours (T): {T_val:.2f}")
        print(f"  Overtime Hours (y1 + 2*y2): {OT_val:.2f}")
else:
    print("No optimal solution found for manual relaxation.")
