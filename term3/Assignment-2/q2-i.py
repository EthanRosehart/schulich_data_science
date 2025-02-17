import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_covering_model_part_i_binary_ot(hotels_csv_path):
    """
    Pure-binary covering model for part (i) using only binary variables to track overtime.
    Model details:
      - 8 attendants.
      - $25/hr base wage → $200 guaranteed day pay if used.
      - If an attendant cleans more than 3500 sq ft, their day pay doubles (+$200).
      - Overtime above 8 hr is paid at 2× base (i.e. $50/hr) for full hours only,
           using three mutually exclusive modes:
           • mode 0: total cleaning time ≤ 8 hr (no OT)
           • mode 1: total cleaning time > 8 and ≤ 9 hr (1 OT hour)
           • mode 2: total cleaning time > 9 and ≤ 10 hr (2 OT hours)
      - If an attendant cleans rooms on more than 2 floors, they receive an extra $75 per extra floor (max 2).
    The model assigns rooms (with known cleaning times, square feet, and floor) to attendants,
    and uses only binary variables to track whether an attendant is used and, if so, in which “mode” (i.e. how many full OT hours are incurred).
    """
    # 1) Read data
    df = pd.read_csv(hotels_csv_path)
    rooms = df["Room_ID"].tolist()
    floor_of = dict(zip(df["Room_ID"], df["Floor"]))
    area_of  = dict(zip(df["Room_ID"], df["Square_Feet"]))
    time_of  = dict(zip(df["Room_ID"], df["Cleaning_Time_Hours"]))
    floors_list = sorted(df["Floor"].unique())

    # 8 attendants
    I = 8
    attendants = range(I)

    # 2) Create model
    m = gp.Model("HotelCovering_Part_i_BinaryOT")

    # 3) Decision Variables
    # x[i,r] = 1 if attendant i cleans room r
    x = {}
    for i in attendants:
        for r in rooms:
            x[(i, r)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")

    # y[i] = 1 if attendant i is used (i.e. cleans at least one room)
    y = {}
    for i in attendants:
        y[i] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    # w[i] = 1 if attendant i cleans more than 3500 sq ft (triggers double day pay)
    w = {}
    for i in attendants:
        w[i] = m.addVar(vtype=GRB.BINARY, name=f"w_{i}")

    # fvar[i,k] = 1 if attendant i cleans any room on floor k
    fvar = {}
    for i in attendants:
        for k in floors_list:
            fvar[(i, k)] = m.addVar(vtype=GRB.BINARY, name=f"f_{i}_{k}")

    # Now, instead of continuous total time T and overtime OT,
    # we use binary “mode” variables to capture the discrete overtime structure.
    # For each attendant i, we define three binary variables:
    #    mode[i,0] = 1 if i works ≤ 8 hr (no OT)
    #    mode[i,1] = 1 if i works > 8 and ≤ 9 hr (1 OT hour)
    #    mode[i,2] = 1 if i works > 9 and ≤ 10 hr (2 OT hours)
    mode = {}
    for i in attendants:
        for k in range(3):
            mode[(i, k)] = m.addVar(vtype=GRB.BINARY, name=f"mode_{i}_{k}")

    # Link the "used" variable with the mode variables.
    # If an attendant is used, exactly one mode must be active.
    for i in attendants:
        m.addConstr(gp.quicksum(mode[(i, k)] for k in range(3)) == y[i],
                    name=f"mode_sum_{i}")

    # Time constraint: the total cleaning time for attendant i (computed from x variables)
    # must be no more than the limit corresponding to the active mode.
    # For mode 0: limit = 8, mode 1: limit = 9, mode 2: limit = 10.
    # (We assume cleaning times are integer—or scaled to integers.)
    for i in attendants:
        m.addConstr(gp.quicksum(time_of[r] * x[(i, r)] for r in rooms)
                    <= 8*mode[(i, 0)] + 9*mode[(i, 1)] + 10*mode[(i, 2)],
                    name=f"time_limit_{i}")

    # (B) Activation constraint: if any room is assigned to i then y[i]=1.
    num_rooms = len(rooms)
    for i in attendants:
        m.addConstr(gp.quicksum(x[(i, r)] for r in rooms) <= num_rooms * y[i],
                    name=f"activate_attendant_{i}")

    # (A) Each room is covered exactly once.
    for r in rooms:
        m.addConstr(gp.quicksum(x[(i, r)] for i in attendants) == 1,
                    name=f"cover_room_{r}")

    # (E) Square footage: if total area cleaned by i exceeds 3500, then w[i]=1.
    # total_area is used as an upper bound.
    total_area = df["Square_Feet"].sum()
    for i in attendants:
        area_cleaned = gp.quicksum(area_of[r] * x[(i, r)] for r in rooms)
        m.addConstr(area_cleaned <= 3500 + total_area * w[i],
                    name=f"area_up_{i}")
        m.addConstr(3500 * w[i] <= area_cleaned,
                    name=f"area_lo_{i}")

    # (F) Floor usage: if i cleans any room on floor k, then fvar[i,k]=1.
    for i in attendants:
        for k in floors_list:
            rooms_k = [r for r in rooms if floor_of[r] == k]
            m.addConstr(gp.quicksum(x[(i, r)] for r in rooms_k)
                        <= len(rooms_k) * fvar[(i, k)],
                        name=f"fvar_{i}_{k}")
    # Define an integer variable for the total number of floors cleaned by i.
    floors_i = {}
    for i in attendants:
        floors_i[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(floors_list),
                               name=f"floors_{i}")
        m.addConstr(floors_i[i] == gp.quicksum(fvar[(i, k)] for k in floors_list),
                    name=f"count_floors_{i}")
        m.addConstr(floors_i[i] <= len(floors_list) * y[i],
                    name=f"maxfloors_{i}")

    # Extra floors: if attendant i cleans on more than 2 floors, then each extra floor (up to 2)
    # incurs a penalty of $75.
    ef = {}
    for i in attendants:
        ef[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=2, name=f"ef_{i}")
        m.addConstr(ef[i] >= floors_i[i] - 2, name=f"ef_ge_{i}")
        m.addConstr(ef[i] <= floors_i[i],     name=f"ef_le_{i}")

    m.update()

    # 5) Objective Function
    # For each attendant i:
    #   - Base day pay: $200 if used.
    #   - Overtime: if mode 1 is active, add $50; if mode 2 is active, add $100.
    #   - If the attendant cleans >3500 sq ft (w[i]=1), add an extra $200 (i.e. day pay doubles).
    #   - Floor penalty: $75 for each extra floor beyond 2.
    cost_expr = gp.LinExpr()
    for i in attendants:
        base_pay = 200 * y[i]
        ot_cost = 50 * mode[(i, 1)] + 100 * mode[(i, 2)]
        sqft_penalty = 200 * w[i]
        floor_penalty = 75 * ef[i]
        cost_expr += base_pay + ot_cost + sqft_penalty + floor_penalty

    m.setObjective(cost_expr, GRB.MINIMIZE)

    # 6) Solve the model
    m.optimize()

    # 7) Print results (in a style similar to our Covering Problem example)
    if m.status == GRB.OPTIMAL:
        print(f"Optimal solution found with total cost = ${m.objVal:,.2f}\n")
        total_OT_hours = 0  # Total overtime hours (0, 1, or 2 per used attendant)
        total_extra_floors = 0
        total_sqft_penalties = 0  # Count of attendants triggering double day pay

        for i in attendants:
            if y[i].X > 0.5:
                assigned_rooms = [r for r in rooms if x[(i, r)].X > 0.5]
                # Determine the mode (only one of these will be 1)
                if mode[(i,0)].X > 0.5:
                    mode_str = "<= 8 hrs (no OT)"
                    ot_hours = 0
                elif mode[(i,1)].X > 0.5:
                    mode_str = "> 8 and <= 9 hrs (1 OT hour)"
                    ot_hours = 1
                elif mode[(i,2)].X > 0.5:
                    mode_str = "> 9 and <= 10 hrs (2 OT hours)"
                    ot_hours = 2
                else:
                    mode_str = "Not used"
                    ot_hours = 0

                total_time = sum(time_of[r] for r in assigned_rooms)
                total_area_cleaned = sum(area_of[r] for r in assigned_rooms)
                fl_used = sum(fvar[(i, k)].X for k in floors_list)
                extra_floors = ef[i].X

                cost_i = 200 * y[i].X + 50 * mode[(i,1)].X + 100 * mode[(i,2)].X + 200 * w[i].X + 75 * ef[i].X

                print(f"Attendant {i}:")
                print(f"  Rooms assigned: {assigned_rooms}")
                print(f"  Total cleaning time = {total_time:.2f} hrs, Area = {total_area_cleaned:.0f} sq ft")
                print(f"  Mode: {mode_str}")
                print(f"  Floors used = {int(fl_used)} (extra floors = {int(extra_floors)})")
                print(f"  Cost breakdown: base=$200, OT addition = ${50 if mode[(i,1)].X>0.5 else 0}" +
                      f" + ${100 if mode[(i,2)].X>0.5 else 0}, " +
                      f"sqft penalty = ${200 if w[i].X>0.5 else 0}, floor penalty = ${75 * extra_floors}")
                print(f"  => Attendant cost = ${cost_i:.2f}\n")

                total_OT_hours += ot_hours
                total_extra_floors += extra_floors
                if w[i].X > 0.5:
                    total_sqft_penalties += 1

        print("=== Summary of Key Penalties ===")
        print(f" Total Overtime Hours (sum across attendants): {total_OT_hours}")
        print(f" Total Extra Floors (sum of floors above 2): {total_extra_floors}")
        print(f" Number of Attendants Exceeding 3500 sq ft: {total_sqft_penalties}")
    else:
        print(f"No optimal solution found. Solver status = {m.status}")

if __name__ == "__main__":
    solve_covering_model_part_i_binary_ot("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/hotels.csv")
