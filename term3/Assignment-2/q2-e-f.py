import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# =========================================================
# 1) DATA LOADING
# =========================================================
url = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/hotels.csv"
df = pd.read_csv(url)

# The dataframe has columns: Room_ID, Floor, Square_Feet, Cleaning_Time_Hours
# We'll create a list of rooms, each with attributes (floor, sq ft, cleaning time)
rooms = df.to_dict("records")  # list of dicts

# We assume 8 attendants
num_attendants = 8
attendants = range(num_attendants)  # 0..7 for convenience

# Wage
regular_wage = 25.0  # $25/hour

# =========================================================
# 2) MODEL SETUP
# =========================================================
m = gp.Model("HotelStaffing")

# ---------------------------------------------------------
# 2a) Decision Variables
# ---------------------------------------------------------

# x[i,r]: 1 if attendant i cleans room r, else 0
x = {}
for i in attendants:
    for r, roominfo in enumerate(rooms):
        x[(i, r)] = m.addVar(vtype=GRB.BINARY,
                             name=f"x_i{i}_room{r}")

# y[i]: 1 if attendant i is employed at all (cleans >=1 room), 0 otherwise
y = {}
for i in attendants:
    y[i] = m.addVar(vtype=GRB.BINARY, name=f"employed_{i}")

# f[i,k]: 1 if attendant i cleans >=1 room on floor k, else 0
# We'll collect floors from the CSV
all_floors = sorted(df["Floor"].unique())
f = {}
for i in attendants:
    for floor_k in all_floors:
        f[(i, floor_k)] = m.addVar(vtype=GRB.BINARY,
                                   name=f"floor_{floor_k}_i{i}")

# Overtime flags:
# z1[i] => attendant i works exactly 1 OT hour
# z2[i] => attendant i works exactly 2 OT hours
z1 = {}
z2 = {}
for i in attendants:
    z1[i] = m.addVar(vtype=GRB.BINARY, name=f"OT1_{i}")
    z2[i] = m.addVar(vtype=GRB.BINARY, name=f"OT2_{i}")

# If double_wage[i] = 1 => attendant i's wage doubles
#    (i.e. they exceed 3500 sq ft)
double_wage = {}
for i in attendants:
    double_wage[i] = m.addVar(vtype=GRB.BINARY, name=f"doubleWage_{i}")

# For 3-or-4-floor penalty:
# b3[i] => attendant i cleans exactly 3 floors
# b4[i] => attendant i cleans exactly 4 floors
b3 = {}
b4 = {}
for i in attendants:
    b3[i] = m.addVar(vtype=GRB.BINARY, name=f"threeFloors_{i}")
    b4[i] = m.addVar(vtype=GRB.BINARY, name=f"fourFloors_{i}")

m.update()

# ---------------------------------------------------------
# 2b) Constraints
# ---------------------------------------------------------

# (i) Each room is assigned to exactly ONE attendant
for r, roominfo in enumerate(rooms):
    m.addConstr(gp.quicksum(x[(i, r)] for i in attendants) == 1,
                name=f"oneAttendantForRoom_{r}")

# (ii) Linking y[i] (employed) to x[i,r]: If x[i,r]=1 for any r => y[i]=1
#      => x[i,r] <= y[i]
for i in attendants:
    for r, roominfo in enumerate(rooms):
        m.addConstr(x[(i, r)] <= y[i], name=f"empLink_i{i}_r{r}")

# (iii) Compute total cleaning hours for each attendant
#       H_i = sum_{r} cleaning_time[r]*x[i,r]
#       We won't create H_i as a variable directly; we can do big-M logic
#       or define H_i as a continuous var and link it. Let's define it:
H = {}
for i in attendants:
    H[i] = m.addVar(lb=0.0, name=f"hours_{i}")

# sum of cleaning times
for i in attendants:
    m.addConstr(H[i] == gp.quicksum(rooms[r]["Cleaning_Time_Hours"] * x[(i, r)]
                                    for r in range(len(rooms))),
                name=f"defHours_{i}")

# (iv) Hours logic with y[i], z1[i], z2[i].
#     - If y[i]=1 => H[i] >= 0 up to 10
#     - z1[i]=1 => H[i]=9, z2[i]=1 => H[i]=10, etc.
#     This can get tricky. One simpler approach:
#         H[i] = 8*y[i] + 1*z1[i] + 2*z2[i]
#     But that forces exactly that formula. We'll allow up to that:
m.addConstrs((H[i] <= 8 + 1*z1[i] + 2*z2[i]
              for i in attendants),
             name="OT_Cap")
# Also, z1[i]+z2[i] <= 1 => can't have both
for i in attendants:
    m.addConstr(z1[i] + z2[i] <= 1, name=f"OT_exclusive_{i}")
# If y[i]=0 => H[i]=0 => z1[i]=z2[i]=0
for i in attendants:
    m.addConstr(H[i] <= 10*y[i], name=f"hoursIfEmployed_{i}")

# (v) Linking floors: f[(i,floor_k)] => 1 if attendant i cleans any room on that floor
for i in attendants:
    for floor_k in all_floors:
        # For each room r on that floor:
        room_indices = [r for r in range(len(rooms)) if rooms[r]["Floor"] == floor_k]
        # x[i,r] <= f[(i,floor_k)] for each r
        for r in room_indices:
            m.addConstr(x[(i, r)] <= f[(i, floor_k)],
                        name=f"floorLink_i{i}_floor{floor_k}_room{r}")

# (vi) Each attendant i can clean between 2 and 4 floors if y[i]=1
# => 2*y[i] <= sum_k f[(i,k)] <= 4*y[i]
for i in attendants:
    m.addConstr(gp.quicksum(f[(i, floor_k)] for floor_k in all_floors) >= 2*y[i],
                name=f"minFloor_{i}")
    m.addConstr(gp.quicksum(f[(i, floor_k)] for floor_k in all_floors) <= 4*y[i],
                name=f"maxFloor_{i}")

# (vii) b3[i], b4[i] => exactly 3 or 4 floors
# Let F_i = sum_k f[(i,k)]. Then:
#  b3[i] = 1 => F_i=3
#  b4[i] = 1 => F_i=4
# plus we can't have both b3[i] and b4[i]
F_i = {}
for i in attendants:
    F_i[i] = m.addVar(lb=0, ub=len(all_floors), name=f"F_i{i}")

    # define F_i
    m.addConstr(F_i[i] == gp.quicksum(f[(i, floor_k)] for floor_k in all_floors),
                name=f"defFi_{i}")

    # big M logic: F_i[i]=3 <=> b3[i]=1
    # => F_i[i] - 3 <=  M*(1 - b3[i])
    # => 3 - F_i[i] <=  M*(1 - b3[i])
    # We'll pick M = 16 or something large enough
    M = 16
    m.addConstr(F_i[i] - 3 <= M*(1 - b3[i]), name=f"b3_1_{i}")
    m.addConstr(3 - F_i[i] <= M*(1 - b3[i]), name=f"b3_2_{i}")

    # same for F_i=4 <=> b4[i]=1
    m.addConstr(F_i[i] - 4 <= M*(1 - b4[i]), name=f"b4_1_{i}")
    m.addConstr(4 - F_i[i] <= M*(1 - b4[i]), name=f"b4_2_{i}")

    # can't both be 1
    m.addConstr(b3[i] + b4[i] <= 1, name=f"no34_{i}")

# (viii) Square footage > 3500 => doubleWage[i]=1
Sq_i = {}
for i in attendants:
    Sq_i[i] = m.addVar(lb=0, name=f"sqft_{i}")
    # define total sq ft
    m.addConstr(Sq_i[i] == gp.quicksum(rooms[r]["Square_Feet"]*x[(i,r)]
                                       for r in range(len(rooms))),
                name=f"defSqft_{i}")
    # "If >3500 => doubleWage[i]=1" => bigM again
    # We'll pick M=99999 or so, or something bigger than any possible sum
    BIG = 99999
    #  Sq_i[i] - 3500 <= BIG*(doubleWage[i])
    #  If doubleWage[i]=0 => Sq_i[i] <= 3500
    m.addConstr(Sq_i[i] - 3500 <= BIG*double_wage[i], name=f"sqDouble_1_{i}")
    # Also if doubleWage[i]=1 => we allow sq ft up to max
    # no contradiction if it's smaller though

# ---------------------------------------------------------
# 2c) Objective Function
# ---------------------------------------------------------
# We'll piece together the cost:
# 1) base cost: 8h of normal wage if y[i]=1 => 8*25*y[i]
# 2) OT cost: 
#    - 1 hour OT => 25*(2.0 - 1.0)=25 extra or 25*(2.0) total?
#    Actually let's do: if z1[i]=1 => 1 OT hour => 1 * (OT rate - normal) = 1*(50-25)=25 if we do double wage? 
#    or 1*(37.5 - 25)=12.5 if 1.5 wage
#    Let's assume 1.5 was the original. We'll do 1.5 => 25 + 12.5 =37.5
#    But user said "Now it's 1.5 or 2.0"? 
# For simplicity, let's assume 1.5 => 1 hour => 25*(1.5 -1)=12.5
#  2 hours => 2*(25*0.5)=25
#
# 3) If double_wage[i]=1 => we double their entire base wage? We'll do a 
#    'double factor' approach. 
# We'll do a single expression that tries to capture it. 
#
# Let's keep it simpler: We'll add the 'double wage penalty' as a lumpsum if double_wage=1
# for an 8hr shift => +200 if they worked 8 hours, + ??? if 10 hours
# It's a bit tricky. We'll do a bigM approach:
#
# 4) 3 floors => +75, 4 floors => +150

cost = gp.LinExpr()

OVERTIME_RATE_DELTA = 25*0.5  # extra 0.5 * 25 = 12.5 per hour
# base cost for 8 hours if y[i]=1 => 8*25=200
for i in attendants:
    # base
    cost.addTerms(200.0, y[i])  # 8 * 25
    # OT1 => 1 hour @ +12.5 
    cost.addTerms(12.5, z1[i])
    # OT2 => 2 hours @ +12.5 each = 25
    cost.addTerms(25.0, z2[i])
    # 3 floors => +75
    cost.addTerms(75.0, b3[i])
    # 4 floors => +150
    cost.addTerms(150.0, b4[i])
    
    # Double wage if sq ft > 3500 => 
    # For simplicity: add a lumpsum "double everything" penalty. 
    # E.g. If H[i] is the total hours, the base pay is (8 + OT) * 25 if y[i]=1. 
    # Doubling that entire pay is an extra "the same amount" again if doubleWage=1. 
    # Let's define an expression for total pay w/o doubling:
    #   pay_i_noDouble = 200*y[i] + 12.5*z1[i] + 25*z2[i]
    # So doubling => we add that again if doubleWage=1
    # => cost += pay_i_noDouble * doubleWage[i]
    # But we can't do cost += pay_i_noDouble * doubleWage[i] directly, 
    #   because that's a product of variable (pay_i_noDouble) and binary. 
    # We'll create a bigM variable or replicate the logic with a separate variable.
    # For demonstration, let's do a single "extraDoublePay_i" and link it via bigM.
    extraDoublePay_i = m.addVar(lb=0.0, name=f"extraDouble_{i}")
    # cost includes that
    cost.add(extraDoublePay_i)
    # Link extraDoublePay_i >= pay_i_noDouble - M*(1 - doubleWage[i])
    # We'll compute pay_i_noDouble as a constant expression:
    payNoDouble = 200.0*y[i] + 12.5*z1[i] + 25.0*z2[i]
    # Gurobi doesn't let us do linear combos with these variable coefs easily, 
    # so let's do: payNoDouble_i = a separate var:
    payNoDouble_i = m.addVar(lb=0.0, name=f"payNoDouble_{i}")
    m.addConstr(payNoDouble_i == 200.0*y[i] + 12.5*z1[i] + 25*z2[i],
                name=f"defPayNoDouble_{i}")
    BIG_2 = 10000
    # extraDoublePay_i >= payNoDouble_i - BIG_2*(1-doubleWage[i])
    # extraDoublePay_i <= payNoDouble_i
    m.addConstr(extraDoublePay_i >= payNoDouble_i - BIG_2*(1-double_wage[i]),
                name=f"doubleLink1_{i}")
    m.addConstr(extraDoublePay_i <= payNoDouble_i, name=f"doubleLink2_{i}")
    m.addConstr(extraDoublePay_i <= BIG_2*double_wage[i],
                name=f"doubleLink3_{i}")

m.setObjective(cost, GRB.MINIMIZE)

# =========================================================
# 3) Solve Model
# =========================================================
m.setParam('MIPGap', 1e-9)
m.optimize()

# =========================================================
# 4) Results
# =========================================================

if m.status == GRB.OPTIMAL:
    print(f"Optimal Cost = {m.objVal:.2f}")
    
    # Let's count total OT hours and total 'floor >2' violations
    total_ot_hours = 0
    total_3floors = 0
    total_4floors = 0
    
    for i in attendants:
        # hours
        hours_i = H[i].X
        # if hours_i>8 => that's OT
        # we can check z1[i].X, z2[i].X
        ot1 = z1[i].X
        ot2 = z2[i].X
        # total OT for this attendant
        if ot1>0.5:  # means 1 hr
            total_ot_hours += 1
        if ot2>0.5:  # means 2 hr
            total_ot_hours += 2
        
        # check floors
        if b3[i].X>0.5:
            total_3floors += 1
        if b4[i].X>0.5:
            total_4floors += 1
    
    print(f"Total Overtime Hours = {int(total_ot_hours)}")
    print(f"Number of Attendants with 3-floor penalty = {total_3floors}")
    print(f"Number of Attendants with 4-floor penalty = {total_4floors}")
    
    # Optionally print assignment
    for i in attendants:
        assigned_rooms = [r for r in range(len(rooms)) if x[(i,r)].X>0.5]
        if assigned_rooms:
            print(f"Attendant {i} assigned rooms {assigned_rooms}, Hours={H[i].X:.1f}")
else:
    print("No optimal solution found.")

# ============ Create and Solve the Relaxed Model ============
relaxed_model = m.relax()  # All integer vars become continuous
relaxed_model.optimize()

print("\n========== Relaxed Model Results ==========")
if relaxed_model.status == GRB.OPTIMAL:
    print(f"Optimal Cost (Relaxation) = {relaxed_model.objVal:.2f}\n")
    
    # Just like you did for the integer solution, you can sum up
    # the (now-fractional) values of certain variables to see how
    # "partially used" they are in the relaxed solution.
    
    sum_employed = 0.0
    sum_ot1 = 0.0
    sum_ot2 = 0.0
    sum_3floors = 0.0
    sum_4floors = 0.0
    sum_double = 0.0
    
    # We'll iterate over the relaxed model's variables and check their names
    # to match them to y[i], z1[i], z2[i], b3[i], b4[i], double_wage[i], etc.
    for v in relaxed_model.getVars():
        vname = v.varName
        if vname.startswith("employed_"):  # y[i]
            sum_employed += v.X
        elif vname.startswith("OT1_"):     # z1[i]
            sum_ot1 += v.X
        elif vname.startswith("OT2_"):     # z2[i]
            sum_ot2 += v.X
        elif vname.startswith("threeFloors_"):  # b3[i]
            sum_3floors += v.X
        elif vname.startswith("fourFloors_"):   # b4[i]
            sum_4floors += v.X
        elif vname.startswith("doubleWage_"):   # double_wage[i]
            sum_double += v.X
    
    # Now print these fractional sums
    print(f"Sum of 'employed' flags (relaxed)   = {sum_employed:.2f}")
    print(f"Sum of OT1 flags (relaxed)          = {sum_ot1:.2f}")
    print(f"Sum of OT2 flags (relaxed)          = {sum_ot2:.2f}")
    print(f"Sum of '3-floor' flags (relaxed)    = {sum_3floors:.2f}")
    print(f"Sum of '4-floor' flags (relaxed)    = {sum_4floors:.2f}")
    print(f"Sum of 'double wage' flags (relaxed)= {sum_double:.2f}")
    
    # (Optionally, you could also check partial room assignments, partial floors, etc.)
else:
    print("Relaxed model not optimal or infeasible.")