#!/usr/bin/env python3
"""
Complete Gurobi model for Question 2 using indicator (binary) constraints.
Assumptions:
 • hotels.csv is available at the specified URL and contains columns:
   Floor, Cleaning_Time_Hours, Square_Feet
 • 8 attendants, each with a base wage of $25/hr.
 • Each attendant is paid for at least 8 hours (base pay) and any work beyond 8 hours,
   up to a maximum of 10 hours, is billed at 1.5× the wage in full-hour increments.
 • An attendant cleaning more than 2 floors incurs a penalty:
       3 floors → +$75; 4 floors → +$150.
"""

import pandas as pd
from gurobipy import Model, GRB, quicksum

# -------------------------
# Data Loading
# -------------------------
rooms_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/hotels.csv')
rooms = list(rooms_df.index)  # each room is indexed by its row number
floors = sorted(rooms_df['Floor'].unique())
# Create dictionaries for room parameters:
cleaning_time = {j: rooms_df.loc[j, 'Cleaning_Time_Hours'] for j in rooms}
Square_Feet   = {j: rooms_df.loc[j, 'Square_Feet'] for j in rooms}
room_floor    = {j: rooms_df.loc[j, 'Floor'] for j in rooms}

# -------------------------
# Parameters
# -------------------------
attendants = range(8)         # 8 attendants indexed 0,...,7
wage       = 25               # hourly wage
base_hours = 8                # base hours per attendant (paid regardless)
max_time   = 10               # maximum cleaning time (8 + up to 2 overtime)
epsilon    = 1e-3            # tolerance for strict inequalities

# -------------------------
# Create Model
# -------------------------
m = Model("HotelWorkerProtection")

# Decision Variables:
# x[i,j] = 1 if attendant i cleans room j.
x = m.addVars(attendants, rooms, vtype=GRB.BINARY, name="x")

# e[i] = 1 if attendant i is employed (assigned at least one room).
e = m.addVars(attendants, vtype=GRB.BINARY, name="e")

# Overtime is discretized into three categories:
# y0[i] = 1 if attendant i works no overtime (T[i] <= 8).
# y1[i] = 1 if attendant i works exactly 1 overtime hour (8 < T[i] <= 9).
# y2[i] = 1 if attendant i works exactly 2 overtime hours (9 < T[i] <= 10).
y0 = m.addVars(attendants, vtype=GRB.BINARY, name="y0")
y1 = m.addVars(attendants, vtype=GRB.BINARY, name="y1")
y2 = m.addVars(attendants, vtype=GRB.BINARY, name="y2")
for i in attendants:
    m.addConstr(y0[i] + y1[i] + y2[i] == e[i], name=f"OT_category_{i}")

# Floor discretization:
# f2[i] = 1 if attendant i cleans exactly 2 floors (minimum, no penalty)
# f3[i] = 1 if attendant i cleans 3 floors (penalty $75)
# f4[i] = 1 if attendant i cleans 4 floors (penalty $150)
f2 = m.addVars(attendants, vtype=GRB.BINARY, name="f2")
f3 = m.addVars(attendants, vtype=GRB.BINARY, name="f3")
f4 = m.addVars(attendants, vtype=GRB.BINARY, name="f4")
for i in attendants:
    m.addConstr(f2[i] + f3[i] + f4[i] == e[i], name=f"Floor_category_{i}")

# For linking floors: a[i,k] = 1 if attendant i cleans any room on floor k.
a = m.addVars(attendants, floors, vtype=GRB.BINARY, name="a")
# F[i] = number of distinct floors attendant i cleans.
F = m.addVars(attendants, vtype=GRB.INTEGER, name="F")

# Continuous variables for total cleaning time and total square footage per attendant.
T = m.addVars(attendants, vtype=GRB.CONTINUOUS, name="T")
S = m.addVars(attendants, vtype=GRB.CONTINUOUS, name="S")

# v[i] = 1 if attendant i exceeds 3500 square feet (and thus gets double base wage).
v = m.addVars(attendants, vtype=GRB.BINARY, name="v")

# -------------------------
# Constraints
# -------------------------
# (1) Each room must be cleaned by exactly one attendant.
for j in rooms:
    m.addConstr(quicksum(x[i, j] for i in attendants) == 1, name=f"Room_{j}_assignment")

# (2) Link employment and room assignments.
for i in attendants:
    for j in rooms:
        m.addGenConstrIndicator(e[i], 0, x[i, j] == 0, name=f"Link_e_{i}_{j}")
    m.addGenConstrIndicator(e[i], 1, quicksum(x[i, j] for j in rooms) >= 1, name=f"Link_e_min_{i}")

# (3) Define total cleaning hours (T) and total square footage (S) per attendant.
for i in attendants:
    m.addConstr(T[i] == quicksum(cleaning_time[j] * x[i, j] for j in rooms), name=f"TotalTime_{i}")
    m.addConstr(S[i] == quicksum(Square_Feet[j] * x[i, j] for j in rooms), name=f"TotalSquare_Feet_{i}")

# (4) Link square footage indicator:
for i in attendants:
    m.addGenConstrIndicator(v[i], 0, S[i] <= 3500, name=f"Square_Feet_indicator_low_{i}")
    m.addGenConstrIndicator(v[i], 1, S[i] >= 3500 + epsilon, name=f"Square_Feet_indicator_high_{i}")

# (5) Overtime category linking via indicator constraints:
# If y0[i] == 1 then T[i] <= base_hours.
# If y1[i] == 1 then base_hours < T[i] <= 9.
# If y2[i] == 1 then 9 < T[i] <= 10.
for i in attendants:
    m.addGenConstrIndicator(y0[i], 1, T[i] <= base_hours, name=f"OT_y0_{i}")
    m.addGenConstrIndicator(y1[i], 1, T[i] >= base_hours + epsilon, name=f"OT_y1_lb_{i}")
    m.addGenConstrIndicator(y1[i], 1, T[i] <= 9, name=f"OT_y1_ub_{i}")
    m.addGenConstrIndicator(y2[i], 1, T[i] >= 9 + epsilon, name=f"OT_y2_lb_{i}")
    m.addGenConstrIndicator(y2[i], 1, T[i] <= 10, name=f"OT_y2_ub_{i}")

# (6) Link room assignments to floor coverage.
for i in attendants:
    for k in floors:
        rooms_on_k = [j for j in rooms if room_floor[j] == k]
        m.addGenConstrIndicator(a[i, k], 0, quicksum(x[i, j] for j in rooms_on_k) == 0, name=f"Link_a_{i}_{k}")
        m.addGenConstrIndicator(a[i, k], 1, quicksum(x[i, j] for j in rooms_on_k) >= 1, name=f"Link_a_on_{i}_{k}")

# (7) Define F[i] = number of floors attendant i cleans.
for i in attendants:
    m.addConstr(F[i] == quicksum(a[i, k] for k in floors), name=f"F_{i}")

# (8) Link floor category with F[i]:
for i in attendants:
    m.addGenConstrIndicator(f2[i], 1, F[i] == 2, name=f"Floor_cat2_{i}")
    m.addGenConstrIndicator(f3[i], 1, F[i] == 3, name=f"Floor_cat3_{i}")
    m.addGenConstrIndicator(f4[i], 1, F[i] == 4, name=f"Floor_cat4_{i}")

# -------------------------
# Objective Function
# -------------------------
# Base cost: each employed attendant is paid for 8 hours at $25/hr.
# (If v[i]==1, the wage doubles; i.e., base cost becomes 8*50 = $400.)
# Overtime cost: each overtime hour is billed at 1.5 * 25 = $37.50.
base_cost_expr = quicksum(e[i] * base_hours * (wage + wage * v[i]) for i in attendants)
overtime_cost_expr = quicksum(y1[i] * (1.5 * wage) + y2[i] * (1.5 * wage * 2) for i in attendants)
# Floor penalty cost: $75 for cleaning 3 floors; $150 for cleaning 4 floors.
floor_penalty_expr = quicksum(f3[i] * 75 + f4[i] * 150 for i in attendants)
m.setObjective(base_cost_expr + overtime_cost_expr + floor_penalty_expr, GRB.MINIMIZE)

m.update()
m.optimize()

print("\n=== Integer Model Results ===")
if m.status == GRB.OPTIMAL:
    print("Integer Model Optimal Total Cost = $%.2f" % m.objVal)
    total_OT = 0
    for i in attendants:
        assigned_rooms = [j for j in rooms if x[i, j].X > 0.5]
        if assigned_rooms:
            # Determine overtime hours: if y1[i] is 1 then 1 overtime hour; if y2[i] is 1 then 2 overtime hours.
            OT_hours = 0
            if y1[i].X > 0.5:
                OT_hours += 1
            if y2[i].X > 0.5:
                OT_hours += 2
            total_OT += OT_hours
            print(f"\nAttendant {i}:")
            print(f"  Rooms Covered: {assigned_rooms}")
            print(f"  Total Cleaning Hours (T): {T[i].X:.2f}")
            print(f"  Overtime Hours: {OT_hours}")
            print(f"  Cost for Attendant: ${200 * e[i].X + 37.5 * OT_hours:.2f}")
        else:
            print(f"\nAttendant {i}: Not employed")
    print(f"\nTotal Overtime Hours = {total_OT}")
else:
    print("No optimal integer solution found.")

# -------------------------
# Relaxed Model
# -------------------------
# The relaxation (m.relax()) removes the integrality restrictions.
# Note: Relaxations of models with indicator constraints may produce degenerate solutions.
relaxed_model = m.relax()
relaxed_model.optimize()

print("\n=== Relaxed Model Results ===")
if relaxed_model.status == GRB.OPTIMAL:
    print("Relaxed Model Optimal Total Cost = $%.2f" % relaxed_model.objVal)
    for i in attendants:
        assigned_rooms_relaxed = [j for j in rooms if relaxed_model.getVarByName(f"x[{i},{j}]").X > 0.5]
        e_val = relaxed_model.getVarByName(f"e[{i}]").X
        T_val = relaxed_model.getVarByName(f"T[{i}]").X
        y1_val = relaxed_model.getVarByName(f"y1[{i}]").X
        y2_val = relaxed_model.getVarByName(f"y2[{i}]").X
        OT_val = y1_val + 2 * y2_val
        print(f"\nAttendant {i}:")
        print(f"  Rooms Covered: {assigned_rooms_relaxed}")
        print(f"  Employed (e): {e_val:.2f}")
        print(f"  Total Cleaning Hours (T): {T_val:.2f}")
        print(f"  Overtime Hours (y1 + 2*y2): {OT_val:.2f}")
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