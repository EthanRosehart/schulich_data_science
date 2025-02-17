#!/usr/bin/env python3
"""
Complete Gurobi model for Question 2 (binary formulation using indicator constraints instead of explicit Big‑M)
Assumptions:
 • hotels.csv exists in the working directory and contains columns: Floor, Cleaning_Time_Hours, Square_Feet
 • There are 8 attendants, each with a base wage of $25/hr.
 • Each attendant must be paid for at least 8 hours; overtime is paid at 1.5× rate.
 • Overtime is allowed in 1‑hour increments (max 2 hours) and the cost is computed based on the category.
 • An attendant cleaning more than 2 floors incurs a penalty:
       3 floors → +$75; 4 floors → +$150.
"""

import pandas as pd
from gurobipy import Model, GRB, quicksum

# -------------------------
# Read data
rooms_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/hotels.csv')
rooms = list(rooms_df.index)  # each room is indexed by its row number
floors = sorted(rooms_df['Floor'].unique())
# Create dictionaries for room parameters:
cleaning_time = {j: rooms_df.loc[j, 'Cleaning_Time_Hours'] for j in rooms}
Square_Feet = {j: rooms_df.loc[j, 'Square_Feet'] for j in rooms}
room_floor = {j: rooms_df.loc[j, 'Floor'] for j in rooms}

# -------------------------
# Parameters
attendants = range(8)            # 8 attendants indexed 0,...,7
wage = 25                        # hourly wage
base_hours = 8                   # base hours per attendant
max_time = 10                    # maximum cleaning time (8 base + up to 2 overtime)
epsilon = 1e-3                   # a small number for strict inequalities

# -------------------------
# Create model
m = Model("HotelWorkerProtection")

# Decision variables:
# x[i,j] = 1 if attendant i cleans room j
x = m.addVars(attendants, rooms, vtype=GRB.BINARY, name="x")

# e[i] = 1 if attendant i is employed (assigned at least one room)
e = m.addVars(attendants, vtype=GRB.BINARY, name="e")

# For each attendant, we “discretize” overtime into three categories:
# y0[i] = 1 if attendant i works no overtime (T[i] <= 8)
# y1[i] = 1 if attendant i works 1 hour overtime (8 < T[i] <= 9)
# y2[i] = 1 if attendant i works 2 hours overtime (9 < T[i] <= 10)
y0 = m.addVars(attendants, vtype=GRB.BINARY, name="y0")
y1 = m.addVars(attendants, vtype=GRB.BINARY, name="y1")
y2 = m.addVars(attendants, vtype=GRB.BINARY, name="y2")
for i in attendants:
    # Exactly one overtime category is chosen if the attendant is employed; if not employed all are 0.
    m.addConstr(y0[i] + y1[i] + y2[i] == e[i], name=f"OT_category_{i}")

# For each attendant, we “discretize” the floor count into three categories:
# f2[i] = 1 if attendant i cleans exactly 2 floors (minimum, no penalty)
# f3[i] = 1 if attendant i cleans 3 floors (penalty $75)
# f4[i] = 1 if attendant i cleans 4 floors (penalty $150)
f2 = m.addVars(attendants, vtype=GRB.BINARY, name="f2")
f3 = m.addVars(attendants, vtype=GRB.BINARY, name="f3")
f4 = m.addVars(attendants, vtype=GRB.BINARY, name="f4")
for i in attendants:
    m.addConstr(f2[i] + f3[i] + f4[i] == e[i], name=f"Floor_category_{i}")

# For linking floors, let a[i,k] = 1 if attendant i cleans any room on floor k.
a = m.addVars(attendants, floors, vtype=GRB.BINARY, name="a")
# F[i] = number of floors attendant i cleans
F = m.addVars(attendants, vtype=GRB.INTEGER, name="F")

# Continuous variables for total cleaning time and square footage for each attendant.
T = m.addVars(attendants, vtype=GRB.CONTINUOUS, name="T")
S = m.addVars(attendants, vtype=GRB.CONTINUOUS, name="S")

# v[i] = 1 if attendant i exceeds 3500 Square_Feet (thus gets double base wage)
v = m.addVars(attendants, vtype=GRB.BINARY, name="v")

# -------------------------
# Constraints

# (1) Each room must be cleaned by exactly one attendant.
for j in rooms:
    m.addConstr(quicksum(x[i, j] for i in attendants) == 1, name=f"Room_{j}_assignment")

# (2) Link employment and room assignments.
# If an attendant is not employed, then no room can be assigned.
for i in attendants:
    for j in rooms:
        # If e[i]==0 then force x[i,j]==0.
        m.addGenConstrIndicator(e[i], 0, x[i, j] == 0, name=f"Link_e_{i}_{j}")
    # Conversely, if e[i]==1 then at least one room must be assigned.
    m.addGenConstrIndicator(e[i], 1, quicksum(x[i, j] for j in rooms) >= 1, name=f"Link_e_min_{i}")

# (3) Define T[i] and S[i] from room assignments.
for i in attendants:
    m.addConstr(T[i] == quicksum(cleaning_time[j] * x[i, j] for j in rooms), name=f"TotalTime_{i}")
    m.addConstr(S[i] == quicksum(Square_Feet[j] * x[i, j] for j in rooms), name=f"TotalSquare_Feet_{i}")

# (4) Link square footage indicator v[i] using indicator constraints:
# If v[i]==0 then S[i] <= 3500; if v[i]==1 then S[i] >= 3500+epsilon.
for i in attendants:
    m.addGenConstrIndicator(v[i], 0, S[i] <= 3500, name=f"Square_Feet_indicator_low_{i}")
    m.addGenConstrIndicator(v[i], 1, S[i] >= 3500 + epsilon, name=f"Square_Feet_indicator_high_{i}")

# (5) Overtime category linking via indicator constraints.
# If y0[i]==1 then T[i] <= base_hours.
# If y1[i]==1 then base_hours < T[i] <= 9.
# If y2[i]==1 then 9 < T[i] <= 10.
for i in attendants:
    m.addGenConstrIndicator(y0[i], 1, T[i] <= base_hours, name=f"OT_y0_{i}")
    m.addGenConstrIndicator(y1[i], 1, T[i] >= base_hours + epsilon, name=f"OT_y1_lb_{i}")
    m.addGenConstrIndicator(y1[i], 1, T[i] <= 9, name=f"OT_y1_ub_{i}")
    m.addGenConstrIndicator(y2[i], 1, T[i] >= 9 + epsilon, name=f"OT_y2_lb_{i}")
    m.addGenConstrIndicator(y2[i], 1, T[i] <= 10, name=f"OT_y2_ub_{i}")

# (6) Link room assignments to floor coverage.
# For each attendant i and floor k, if a[i,k]==0 then no room on floor k is assigned;
# if a[i,k]==1 then at least one room is assigned.
for i in attendants:
    for k in floors:
        rooms_on_k = [j for j in rooms if room_floor[j] == k]
        m.addGenConstrIndicator(a[i, k], 0, quicksum(x[i, j] for j in rooms_on_k) == 0, name=f"Link_a_{i}_{k}")
        m.addGenConstrIndicator(a[i, k], 1, quicksum(x[i, j] for j in rooms_on_k) >= 1, name=f"Link_a_on_{i}_{k}")

# (7) Define F[i] = total floors cleaned by attendant i.
for i in attendants:
    m.addConstr(F[i] == quicksum(a[i, k] for k in floors), name=f"F_{i}")

# (8) Link floor category with F[i] via indicator constraints:
# If f2[i]==1 then F[i] must equal 2; if f3[i]==1 then F[i] must equal 3; if f4[i]==1 then F[i] must equal 4.
for i in attendants:
    m.addGenConstrIndicator(f2[i], 1, F[i] == 2, name=f"Floor_cat2_{i}")
    m.addGenConstrIndicator(f3[i], 1, F[i] == 3, name=f"Floor_cat3_{i}")
    m.addGenConstrIndicator(f4[i], 1, F[i] == 4, name=f"Floor_cat4_{i}")

# -------------------------
# Objective: Minimize total staffing cost
# Base cost: for each employed attendant, 8 hours at either regular wage or double wage
# (We model double wage when v[i]==1; note that (wage + wage*v[i]) gives wage if v[i]==0 and 2*wage if v[i]==1.)
base_cost_expr = quicksum(e[i] * base_hours * (wage + wage * v[i]) for i in attendants)

# Overtime cost: if 1-hour overtime then cost = 1.5*wage; if 2-hour overtime then cost = 2*1.5*wage.
overtime_cost_expr = quicksum(y1[i] * (1.5 * wage) + y2[i] * (1.5 * wage * 2) for i in attendants)

# Floor penalty cost: if cleaning 3 floors add $75; if cleaning 4 floors add $150.
floor_penalty_expr = quicksum(f3[i] * 75 + f4[i] * 150 for i in attendants)

m.setObjective(base_cost_expr + overtime_cost_expr + floor_penalty_expr, GRB.MINIMIZE)

# -------------------------
# Optimize model
m.optimize()

# -------------------------
# Report results
if m.status == GRB.OPTIMAL:
    print("Optimal total cost: $%.2f" % m.objVal)
    total_overtime_hours = sum(y1[i].X * 1 + y2[i].X * 2 for i in attendants)
    total_floor_penalties = sum(f3[i].X + f4[i].X for i in attendants)
    print("Total overtime hours: %.0f" % total_overtime_hours)
    print("Total floor penalty count: %.0f" % total_floor_penalties)
else:
    print("No optimal solution found.")
