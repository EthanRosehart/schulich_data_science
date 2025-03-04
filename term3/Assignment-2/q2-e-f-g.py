import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# ---------------------------
# Part e: Binary Model (Original)
# ---------------------------
def solve_covering_model_binary(hotels_csv_path):
    """
    Binary covering model using the pure-binary approach.
    Tracks:
      - Base day pay ($200 if used)
      - Double day pay if area cleaned > 3500 (w[i]=1 adds extra $200)
      - Overtime using step binaries (b1, b2) so that OT is $37.50 per full OT hour (0,1,or2)
      - Floor penalty: $75 for each extra floor beyond 2 (max 2)
    """
    df = pd.read_csv(hotels_csv_path)
    rooms = df["Room_ID"].tolist()
    floor_of = dict(zip(df["Room_ID"], df["Floor"]))
    area_of  = dict(zip(df["Room_ID"], df["Square_Feet"]))
    time_of  = dict(zip(df["Room_ID"], df["Cleaning_Time_Hours"]))
    distinct_floors = sorted(df["Floor"].unique())

    I = 8
    attendants = range(I)

    m = gp.Model("HotelCovering_Binary")

    # Assignment variables: x[i,r] = 1 if attendant i cleans room r.
    x = {}
    for i in attendants:
        for r in rooms:
            x[(i, r)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")

    # Attendant usage: y[i] = 1 if attendant i is used.
    y = { i: m.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in attendants }

    # Square footage indicator: w[i] = 1 if attendant i cleans >3500 sq ft.
    w = { i: m.addVar(vtype=GRB.BINARY, name=f"w_{i}") for i in attendants }

    # Floor assignment: fvar[i,k] = 1 if attendant i cleans any room on floor k.
    fvar = {}
    for i in attendants:
        for k in distinct_floors:
            fvar[(i, k)] = m.addVar(vtype=GRB.BINARY, name=f"f_{i}_{k}")

    # Total cleaning time: T[i] (continuous but with tight bounds).
    T = { i: m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"T_{i}") for i in attendants }

    # Number of floors cleaned by attendant i.
    floors_i = { i: m.addVar(vtype=GRB.INTEGER, lb=0, ub=4, name=f"floors_{i}") for i in attendants }

    # Extra floors beyond 2 (integer 0..2).
    ef = { i: m.addVar(vtype=GRB.INTEGER, lb=0, ub=2, name=f"ef_{i}") for i in attendants }

    # Binary step variables for overtime:
    # b1[i] = 1 if T[i] > 8; b2[i] = 1 if T[i] > 9.
    b1 = { i: m.addVar(vtype=GRB.BINARY, name=f"b1_{i}") for i in attendants }
    b2 = { i: m.addVar(vtype=GRB.BINARY, name=f"b2_{i}") for i in attendants }
    # Paid overtime hours: oh[i] = b1[i] + b2[i] (so in {0,1,2}).
    oh = { i: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2, name=f"oh_{i}") for i in attendants }

    m.update()

    # -- Constraints --

    # (A) Each room is covered exactly once.
    for r in rooms:
        m.addConstr(gp.quicksum(x[(i, r)] for i in attendants) == 1, name=f"cover_{r}")

    # (B) If any room is assigned to i then y[i] = 1.
    nRooms = len(rooms)
    for i in attendants:
        m.addConstr(gp.quicksum(x[(i, r)] for r in rooms) <= nRooms * y[i], name=f"activate_{i}")

    # (C) Total cleaning time T[i] equals the sum of room cleaning times.
    for i in attendants:
        m.addConstr(T[i] == gp.quicksum(time_of[r] * x[(i, r)] for r in rooms), name=f"T_sum_{i}")
        m.addConstr(T[i] <= 10 * y[i], name=f"T_max_{i}")

    # (D) Square footage: if total area cleaned >3500 then w[i]=1.
    total_area = df["Square_Feet"].sum()
    for i in attendants:
        area_cleaned = gp.quicksum(area_of[r] * x[(i, r)] for r in rooms)
        m.addConstr(area_cleaned <= 3500 + total_area * w[i], name=f"area_up_{i}")
        m.addConstr(3500 * w[i] <= area_cleaned, name=f"area_lo_{i}")

    # (E) Floor usage: if any room on floor k is assigned to i then fvar[i,k]=1.
    for i in attendants:
        for k in distinct_floors:
            rooms_k = [r for r in rooms if floor_of[r] == k]
            m.addConstr(gp.quicksum(x[(i, r)] for r in rooms_k)
                        <= len(rooms_k) * fvar[(i, k)],
                        name=f"floor_{i}_{k}")
        m.addConstr(floors_i[i] == gp.quicksum(fvar[(i, k)] for k in distinct_floors), name=f"floorCount_{i}")
        m.addConstr(floors_i[i] <= 4 * y[i], name=f"maxFloors_{i}")
        m.addConstr(ef[i] >= floors_i[i] - 2, name=f"ef_ge_{i}")
        m.addConstr(ef[i] <= floors_i[i],     name=f"ef_le_{i}")

    # (F) Overtime step constraints.
    for i in attendants:
        m.addConstr(T[i] - 8 <= 2 * b1[i], name=f"b1_link_{i}")
        m.addConstr(T[i] - 9 <= 1 * b2[i], name=f"b2_link_{i}")
        m.addConstr(oh[i] == b1[i] + b2[i], name=f"oh_link_{i}")

    # -- Objective --
    # For each attendant i:
    #   Base day pay = $200 * y[i]
    #   Overtime pay = $37.50 * oh[i]
    #   If area >3500 then extra $200 (i.e. double day pay)
    #   Floor penalty = $75 * ef[i]
    cost_expr = gp.LinExpr()
    for i in attendants:
        cost_expr += 200 * y[i] + 37.5 * oh[i] + 200 * w[i] + 75 * ef[i]
    m.setObjective(cost_expr, GRB.MINIMIZE)

    m.update()
    m.optimize()

    total_cost = m.objVal
    print("\n--- Binary Model Solution (Part e) ---")
    print(f"Optimal solution found with total cost = ${total_cost:,.2f}\n")
    if m.status == GRB.OPTIMAL:
        for i in attendants:
            if y[i].X > 0.5:
                assigned = [r for r in rooms if x[(i, r)].X > 0.5]
                tot_time = T[i].X
                tot_area = sum(area_of[r] for r in assigned)
                fl_used = floors_i[i].X
                extra_fl = ef[i].X
                ot_hours = oh[i].X
                if ot_hours < 0.5:
                    mode_str = "<= 8 hrs (no OT)"
                elif ot_hours < 1.5:
                    mode_str = ">8 and <=9 hrs (1 OT hour)"
                else:
                    mode_str = ">9 and <=10 hrs (2 OT hours)"
                cost_i = 200 + 37.5 * ot_hours + (200 if w[i].X > 0.5 else 0) + 75 * extra_fl
                print(f"Attendant {i}:")
                print(f"  Rooms assigned: {assigned}")
                print(f"  Total cleaning time = {tot_time:.2f} hrs, Area = {tot_area:.0f} sq ft")
                print(f"  OT mode: {mode_str}")
                print(f"  Floors used = {int(fl_used)} (extra floors = {int(extra_fl)})")
                print(f"  Cost = ${cost_i:.2f}\n")
    else:
        print("No optimal solution found in binary model.")
    return m

# ---------------------------
# Part f: .relax() Version
# ---------------------------
def solve_covering_model_relax(hotels_csv_path):
    """
    Solve the binary model and then relax it using m.relax().
    Print the optimal solution from the relaxed model.
    """
    print("\n==============================")
    print("Part f: Relaxed Model (using m.relax())")
    print("==============================\n")
    m_binary = solve_covering_model_binary(hotels_csv_path)
    m_relaxed = m_binary.relax()
    m_relaxed.setParam("OutputFlag", 0)  # suppress solver output
    m_relaxed.optimize()
    total_cost_relaxed = m_relaxed.objVal
    print(f"\n--- .relax() Model Solution ---")
    print(f"Optimal solution from relaxed model with total cost = ${total_cost_relaxed:,.2f}\n")
    if m_relaxed.status == GRB.OPTIMAL:
        for i in range(8):
            y_val = m_relaxed.getVarByName(f"y_{i}").X
            if y_val > 0.5:
                T_val = m_relaxed.getVarByName(f"T_{i}").X
                oh_val = m_relaxed.getVarByName(f"oh_{i}").X
                floors_val = m_relaxed.getVarByName(f"floors_{i}").X
                ef_val = m_relaxed.getVarByName(f"ef_{i}").X
                print(f"Attendant {i}: y = {y_val:.2f}, T = {T_val:.2f} hrs, OT = {oh_val:.2f}, Floors = {floors_val:.2f}, Extra Floors = {ef_val:.2f}")
    else:
        print("No optimal solution found in relaxed model.")
    return m_relaxed

# ---------------------------
# Part g: Manual Relaxation (Converting binary/integer vars to continuous)
# ---------------------------
def solve_covering_model_manual_relax(hotels_csv_path):
    """
    Build the same covering model as in Part e but manually relax integrality by
    converting all binary/integer variables to CONTINUOUS with the same bounds.
    Print the optimal solution from this manually relaxed model.
    """
    print("\n==============================")
    print("Part g: Manual Relaxation (binary vars as continuous)")
    print("==============================\n")
    df = pd.read_csv(hotels_csv_path)
    rooms = df["Room_ID"].tolist()
    floor_of = dict(zip(df["Room_ID"], df["Floor"]))
    area_of  = dict(zip(df["Room_ID"], df["Square_Feet"]))
    time_of  = dict(zip(df["Room_ID"], df["Cleaning_Time_Hours"]))
    distinct_floors = sorted(df["Floor"].unique())

    I = 8
    attendants = range(I)

    m = gp.Model("HotelCovering_ManualRelax")

    # For binary variables, use CONTINUOUS with lb=0 and ub=1.
    x = {}
    for i in attendants:
        for r in rooms:
            x[(i, r)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"x_{i}_{r}")

    y = { i: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"y_{i}") for i in attendants }
    w = { i: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"w_{i}") for i in attendants }
    fvar = {}
    for i in attendants:
        for k in distinct_floors:
            fvar[(i, k)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"f_{i}_{k}")

    T = { i: m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"T_{i}") for i in attendants }
    floors_i = { i: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=4, name=f"floors_{i}") for i in attendants }
    ef = { i: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2, name=f"ef_{i}") for i in attendants }

    # For overtime step variables, convert binary to continuous.
    b1 = { i: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"b1_{i}") for i in attendants }
    b2 = { i: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"b2_{i}") for i in attendants }
    oh = { i: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2, name=f"oh_{i}") for i in attendants }

    m.update()

    # Constraints (same as in binary model)
    for r in rooms:
        m.addConstr(gp.quicksum(x[(i, r)] for i in attendants) == 1, name=f"cover_{r}")

    nRooms = len(rooms)
    for i in attendants:
        m.addConstr(gp.quicksum(x[(i, r)] for r in rooms) <= nRooms * y[i], name=f"activate_{i}")

    for i in attendants:
        m.addConstr(T[i] == gp.quicksum(time_of[r] * x[(i, r)] for r in rooms), name=f"T_sum_{i}")
        m.addConstr(T[i] <= 10 * y[i], name=f"T_max_{i}")

    total_area = df["Square_Feet"].sum()
    for i in attendants:
        area_cleaned = gp.quicksum(area_of[r] * x[(i, r)] for r in rooms)
        m.addConstr(area_cleaned <= 3500 + total_area * w[i], name=f"area_up_{i}")
        m.addConstr(3500 * w[i] <= area_cleaned, name=f"area_lo_{i}")

    for i in attendants:
        for k in distinct_floors:
            rooms_k = [r for r in rooms if floor_of[r] == k]
            m.addConstr(gp.quicksum(x[(i, r)] for r in rooms_k) <= len(rooms_k) * fvar[(i, k)],
                        name=f"floor_{i}_{k}")
        m.addConstr(floors_i[i] == gp.quicksum(fvar[(i, k)] for k in distinct_floors), name=f"floorCount_{i}")
        m.addConstr(floors_i[i] <= 4 * y[i], name=f"maxFloors_{i}")
        m.addConstr(ef[i] >= floors_i[i] - 2, name=f"ef_ge_{i}")
        m.addConstr(ef[i] <= floors_i[i],     name=f"ef_le_{i}")

    for i in attendants:
        m.addConstr(T[i] - 8 <= 2 * b1[i], name=f"b1_link_{i}")
        m.addConstr(T[i] - 9 <= 1 * b2[i], name=f"b2_link_{i}")
        m.addConstr(oh[i] == b1[i] + b2[i], name=f"oh_link_{i}")

    cost_expr = gp.LinExpr()
    for i in attendants:
        cost_expr += 200 * y[i] + 37.5 * oh[i] + 200 * w[i] + 75 * ef[i]
    m.setObjective(cost_expr, GRB.MINIMIZE)

    m.update()
    m.optimize()

    total_cost_manual = m.objVal
    print("\n--- Manual Relaxed Model Solution (Part g) ---")
    if m.status == GRB.OPTIMAL:
        print(f"Optimal solution found with total cost = ${total_cost_manual:,.2f}\n")
        for i in attendants:
            if y[i].X > 0.5:
                assigned = [r for r in rooms if x[(i, r)].X > 0.5]
                tot_time = T[i].X
                tot_area = sum(area_of[r] for r in assigned)
                fl_used = floors_i[i].X
                extra_fl = ef[i].X
                ot_hours = oh[i].X
                if ot_hours < 0.5:
                    mode_str = "<= 8 hrs (no OT)"
                elif ot_hours < 1.5:
                    mode_str = ">8 and <=9 hrs (1 OT hour)"
                else:
                    mode_str = ">9 and <=10 hrs (2 OT hours)"
                cost_i = 200 + 37.5 * ot_hours + (200 if w[i].X > 0.5 else 0) + 75 * extra_fl
                print(f"Attendant {i}:")
                print(f"  Rooms assigned: {assigned}")
                print(f"  Total cleaning time = {tot_time:.2f} hrs, Area = {tot_area:.0f} sq ft")
                print(f"  OT mode: {mode_str}")
                print(f"  Floors used = {int(fl_used)} (extra floors = {int(extra_fl)})")
                print(f"  Cost = ${cost_i:.2f}\n")
    else:
        print("No optimal solution found in manual relaxation.")
    return m

# ---------------------------
# Main: Run all three versions and print solutions
# ---------------------------
if __name__ == "__main__":
    csv_url = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/hotels.csv"
    
    print("============================================")
    print("Part e: Binary Model")
    print("============================================")
    m_binary = solve_covering_model_binary(csv_url)
    
    print("============================================")
    print("Part f: Relaxed Model (using m.relax())")
    print("============================================")
    m_relaxed = m_binary.relax()
    m_relaxed.setParam("OutputFlag", 0)
    m_relaxed.optimize()
    total_cost_relaxed = m_relaxed.objVal
    print("\n--- .relax() Model Optimal Solution ---")
    print(f"Optimal solution from relaxed model with total cost = ${total_cost_relaxed:,.2f}\n")
    if m_relaxed.status == GRB.OPTIMAL:
        for i in range(8):
            y_val = m_relaxed.getVarByName(f"y_{i}").X
            if y_val > 0.5:
                T_val = m_relaxed.getVarByName(f"T_{i}").X
                oh_val = m_relaxed.getVarByName(f"oh_{i}").X
                floors_val = m_relaxed.getVarByName(f"floors_{i}").X
                ef_val = m_relaxed.getVarByName(f"ef_{i}").X
                print(f"Attendant {i}: y = {y_val:.2f}, T = {T_val:.2f} hrs, OT = {oh_val:.2f}, Floors = {floors_val:.2f}, Extra Floors = {ef_val:.2f}")
    else:
        print("No optimal solution found in relaxed model.")
    
    print("============================================")
    print("Part g: Manual Relaxation (binary vars as continuous)")
    print("============================================")
    solve_covering_model_manual_relax(csv_url)
