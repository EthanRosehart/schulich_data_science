#!/usr/bin/env python3
"""
Complete Gurobi model for Question 2 using indicator (binary) constraints.
Assumptions:
 • hotels.csv is available at the specified URL and contains columns:
   Floor, Cleaning_Time_Hours, Square_Feet
 • 8 attendants, each with a base wage of $25/hr.
 • Each attendant is paid for at least 8 hours (base pay) and any work beyond 8 hours,
   up to a maximum of 10 hours, is billed in full-hour increments.
 • Overtime is now paid at double wage, but only the extra premium is charged:
   That is, for each overtime hour, an extra $50 is incurred.
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

# Overtime variables: 
# OT1[i] = 1 if attendant i works at least one overtime hour (i.e. T > 8)
# OT2[i] = 1 if attendant i works at least two overtime hours (i.e. T > 9)
OT1 = m.addVars(attendants, vtype=GRB.BINARY, name="OT1")
OT2 = m.addVars(attendants, vtype=GRB.BINARY, name="OT2")
# No mutual exclusivity constraint is imposed—these are independent decisions.
# (They will be forced by indicator constraints based on T.)

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
    m.addConstr(T[i] <= max_time * e[i], name=f"MaxTime_{i}")  # Ensure T is 0 if not employed and <= 10 if employed.

# (4) Link square footage indicator:
for i in attendants:
    m.addGenConstrIndicator(v[i], 0, S[i] <= 3500, name=f"Square_Feet_indicator_low_{i}")
    m.addGenConstrIndicator(v[i], 1, S[i] >= 3500 + epsilon, name=f"Square_Feet_indicator_high_{i}")

# (5) Overtime indicator constraints:
# For each attendant, if T[i] > 8 then OT1[i] = 1, else OT1[i] = 0.
# And if T[i] > 9 then OT2[i] = 1, else OT2[i] = 0.
for i in attendants:
    m.addGenConstrIndicator(OT1[i], 1, T[i] >= base_hours + epsilon, name=f"OT1_lb_{i}")
    m.addGenConstrIndicator(OT1[i], 0, T[i] <= base_hours, name=f"OT1_ub_{i}")
    m.addGenConstrIndicator(OT2[i], 1, T[i] >= 9 + epsilon, name=f"OT2_lb_{i}")
    m.addGenConstrIndicator(OT2[i], 0, T[i] <= 9, name=f"OT2_ub_{i}")

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

# Stopping constraint

# Set a time limit of 600 seconds (2 minutes)
m.setParam('TimeLimit', 120)
# Set a MIP gap of 1% (0.01)
m.setParam('MIPGap', 0.01)

# Base cost: each employed attendant is paid for 8 hours at $25/hr.
# (If v[i]==1, wage doubles: 8*50 = $400.)
base_cost_expr = quicksum(e[i] * base_hours * (wage + wage * v[i]) for i in attendants)
# Overtime cost: now each overtime hour is billed at an extra $50.
# Total overtime premium = 50*(OT1[i] + OT2[i]).
overtime_cost_expr = quicksum(50 * (OT1[i] + OT2[i]) for i in attendants)
# Floor penalty cost: $75 for cleaning 3 floors; $150 for cleaning 4 floors.
floor_penalty_expr = quicksum(f3[i] * 75 + f4[i] * 150 for i in attendants)
m.setObjective(base_cost_expr + overtime_cost_expr + floor_penalty_expr, GRB.MINIMIZE)

m.update()
m.optimize()

print("\n=== Integer Model Results ===")
if m.status == GRB.OPTIMAL:
    print("Integer Model Optimal Total Cost = $%.2f" % m.objVal)
    total_OT = 0
    total_floor_violations = 0
    for i in attendants:
        assigned_rooms = [j for j in rooms if x[i, j].X > 0.5]
        if assigned_rooms:
            OT_hours = 0
            if OT1[i].X > 0.5:
                OT_hours += 1
            if OT2[i].X > 0.5:
                OT_hours += 1
            total_OT += OT_hours
            violation = 0
            if f3[i].X > 0.5:
                violation = 1
            elif f4[i].X > 0.5:
                violation = 2
            total_floor_violations += violation
            # Now compute the cost:
            # Base cost: $200 if not double wage, $400 if double.
            base_cost = 200 * e[i].X if v[i].X < 0.5 else 400 * e[i].X
            overtime_cost = 50 * OT_hours  # $50 extra per overtime hour.
            floor_cost = 75 * f3[i].X + 150 * f4[i].X
            cost_i = base_cost + overtime_cost + floor_cost
            print(f"\nAttendant {i}:")
            print(f"  Rooms Covered: {assigned_rooms}")
            print(f"  Total Cleaning Hours (T): {T[i].X:.2f}")
            print(f"  Overtime Hours: {OT_hours}")
            print(f"  Cost for Attendant: ${cost_i:.2f}")
            print(f"  Floor Violations: {violation}")
        else:
            print(f"\nAttendant {i}: Not employed")
    print(f"\nTotal Overtime Hours = {total_OT}")
    print(f"Total Floor Violations (in excess of 2 floors) = {total_floor_violations}")
else:
    print("No optimal integer solution found.")



# -------------------------
# Relaxed Model
# -------------------------
relaxed_model = m.relax()
relaxed_model.optimize()

print("\n=== Relaxed Model Results ===")
if relaxed_model.status == GRB.OPTIMAL:
    print("Relaxed Model Optimal Total Cost = $%.2f" % relaxed_model.objVal)
    for i in attendants:
        assigned_rooms_relaxed = [j for j in rooms if relaxed_model.getVarByName(f"x[{i},{j}]").X > 0.5]
        e_val = relaxed_model.getVarByName(f"e[{i}]").X
        T_val = relaxed_model.getVarByName(f"T[{i}]").X
        OT1_val = relaxed_model.getVarByName(f"OT1[{i}]").X
        OT2_val = relaxed_model.getVarByName(f"OT2[{i}]").X
        OT_val = OT1_val + OT2_val
        print(f"\nAttendant {i}:")
        print(f"  Rooms Covered: {assigned_rooms_relaxed}")
        print(f"  Employed (e): {e_val:.2f}")
        print(f"  Total Cleaning Hours (T): {T_val:.2f}")
        print(f"  Overtime Hours (OT1 + OT2): {OT_val:.2f}")
else:
    print("No optimal relaxed solution found.")
