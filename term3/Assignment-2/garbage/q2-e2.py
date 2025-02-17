import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_covering_model_whole_OT(hotels_csv_path):
    """
    Solve the hotel-attendant covering problem where:
      - Each attendant is paid for a FULL extra hour once they exceed 8 hours,
        and for 2 FULL extra hours once they exceed 9 hours (capped at 10 total).
      - The rest of the cost structure matches the assignment:
           * 8 hours guaranteed pay at $25 => $200 if used at all
           * If total area > 3500 => base day pay doubles => +$200
           * Overtime: $37.50 per OT hour (0, 1, or 2) with no fractions
           * Additional $75 for each floor beyond 2 (up to 2 extra floors)
    """

    # --- 1) Read data from the CSV ---
    df = pd.read_csv(hotels_csv_path)
    # We assume columns: Room_ID, Floor, Square_Feet, Cleaning_Time_Hours
    rooms = df["Room_ID"].tolist()
    floor_of = dict(zip(df["Room_ID"], df["Floor"]))
    area_of  = dict(zip(df["Room_ID"], df["Square_Feet"]))
    time_of  = dict(zip(df["Room_ID"], df["Cleaning_Time_Hours"]))
    distinct_floors = sorted(df["Floor"].unique())

    # Number of attendants
    I = 8
    attendants = range(I)

    # --- 2) Create the model ---
    m = gp.Model("HotelCoveringWholeOT")

    # --- 3) Decision Variables ---

    # x[i,r] => 1 if attendant i cleans room r
    x = {}
    for i in attendants:
        for r in rooms:
            x[(i, r)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")

    # y[i] => 1 if attendant i works (cleans ≥1 room)
    y = {}
    for i in attendants:
        y[i] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    # w[i] => 1 if attendant i cleans > 3500 sq ft
    w = {}
    for i in attendants:
        w[i] = m.addVar(vtype=GRB.BINARY, name=f"w_{i}")

    # fvar[i,k] => 1 if attendant i cleans any room on floor k
    fvar = {}
    for i in attendants:
        for k in distinct_floors:
            fvar[(i, k)] = m.addVar(vtype=GRB.BINARY, name=f"f_{i}_{k}")

    # T[i] => total cleaning time (continuous)
    T = {}
    for i in attendants:
        T[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"T_{i}")

    # floors_i[i] => how many floors attendant i cleans
    floors_i = {}
    for i in attendants:
        floors_i[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=4, name=f"floors_{i}")

    # ef[i] => # of extra floors above 2 (integer in [0..2])
    ef = {}
    for i in attendants:
        ef[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=2, name=f"ef_{i}")

    # b1[i], b2[i] => step binaries for paying OT hours as whole hours
    # b1[i] = 1 if T[i] > 8, else 0
    # b2[i] = 1 if T[i] > 9, else 0
    b1 = {}
    b2 = {}
    for i in attendants:
        b1[i] = m.addVar(vtype=GRB.BINARY, name=f"b1_{i}")
        b2[i] = m.addVar(vtype=GRB.BINARY, name=f"b2_{i}")

    # oh[i] => how many OT hours are paid (0, 1, or 2). We'll make oh[i] continuous,
    # but the constraints with b1[i], b2[i] force it to be an integer 0..2
    oh = {}
    for i in attendants:
        oh[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2, name=f"ohours_{i}")

    m.update()

    # --- 4) Constraints ---

    # (A) Each room is covered exactly once
    for r in rooms:
        m.addConstr(
            gp.quicksum(x[(i, r)] for i in attendants) == 1,
            name=f"cover_room_{r}"
        )

    # (B) If x[i,r]≥1 for any r, then y[i]=1
    nRooms = len(rooms)
    for i in attendants:
        m.addConstr(
            gp.quicksum(x[(i, r)] for r in rooms) <= nRooms * y[i],
            name=f"use_attendant_{i}"
        )

    # (C) T[i] => total cleaning time
    for i in attendants:
        m.addConstr(
            T[i] == gp.quicksum(time_of[r] * x[(i, r)] for r in rooms),
            name=f"time_sum_{i}"
        )
        # Max 10 hr total if used
        m.addConstr(T[i] <= 10 * y[i], name=f"max_10hr_{i}")

    # (D) Square footage => if >3500 => w[i]=1
    total_area = df["Square_Feet"].sum()
    for i in attendants:
        cleaned_area = gp.quicksum(area_of[r] * x[(i, r)] for r in rooms)
        m.addConstr(cleaned_area <= 3500 + total_area * w[i],
                    name=f"area_upper_{i}")
        m.addConstr(3500 * w[i] <= cleaned_area,
                    name=f"area_lower_{i}")

    # (E) Floor usage: sum of floors
    for i in attendants:
        # if any room on floor k => fvar[i,k]=1
        for k in distinct_floors:
            rooms_on_k = [r for r in rooms if floor_of[r] == k]
            m.addConstr(
                gp.quicksum(x[(i, r)] for r in rooms_on_k)
                <= len(rooms_on_k) * fvar[(i, k)],
                name=f"floor_use_{i}_{k}"
            )
        # sum fvar => floors_i[i]
        m.addConstr(
            floors_i[i] == gp.quicksum(fvar[(i, k)] for k in distinct_floors),
            name=f"count_floors_{i}"
        )
        # can’t exceed 4 if used
        m.addConstr(floors_i[i] <= 4 * y[i], name=f"max_floors_{i}")

        # ef[i] => floors above 2, up to 2
        m.addConstr(ef[i] >= floors_i[i] - 2, name=f"ef_ge_{i}")
        m.addConstr(ef[i] <= floors_i[i],     name=f"ef_le_{i}")

    # (F) Overtime step constraints
    # b1[i] = 1 if T[i]>8, else 0.  Because T[i] up to 10, we use M=2
    for i in attendants:
        # T[i] - 8 <= 2*b1[i]
        # If T[i] <=8 => T[i]-8 <=0 => b1[i] can be 0.
        # If T[i]>8 => T[i]-8>0 => must have b1[i]=1
        m.addConstr(T[i] - 8 <= 2 * b1[i], name=f"b1_{i}")

        # b2[i] = 1 if T[i]>9, else 0. We'll use M=1
        # T[i] - 9 <= 1*b2[i]
        # If T[i]<=9 => T[i]-9<=0 => b2[i] can be 0
        # If T[i]>9 => T[i]-9>0 => must have b2[i]=1
        m.addConstr(T[i] - 9 <= 1 * b2[i], name=f"b2_{i}")

        # oh[i] = b1[i] + b2[i] (this ensures oh[i] in {0,1,2})
        m.addConstr(oh[i] == b1[i] + b2[i], name=f"ohours_{i}")

    # ------------------------------------------------------------------
    # 5) Objective: 
    # cost_i = 
    #   day-pay = $200 if used,
    #   + another $200 if w[i]=1 (exceeds 3500 sq ft),
    #   + overtime = 37.5 * oh[i]  (0,1,2 full hours),
    #   + floors premium = 75 * ef[i].
    # ------------------------------------------------------------------
    cost_expr = gp.LinExpr()
    for i in attendants:
        day_pay = 200 * y[i]
        double_pay = 200 * w[i]
        ot_cost = 37.5 * oh[i]
        floor_cost = 75 * ef[i]
        cost_expr += (day_pay + double_pay + ot_cost + floor_cost)

    m.setObjective(cost_expr, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # 6) Solve
    # ------------------------------------------------------------------
    m.optimize()

    # ------------------------------------------------------------------
    # 7) Report Results
    # ------------------------------------------------------------------
    if m.status == GRB.OPTIMAL:
        print(f"Optimal solution found with total cost = {m.objVal:,.2f}\n")
        for i in attendants:
            if y[i].X > 0.5:  # used
                assigned_rooms = [r for r in rooms if x[(i, r)].X > 0.5]
                tot_time = T[i].X
                tot_area = sum(area_of[r] for r in assigned_rooms)
                flo_count = floors_i[i].X
                extra_flo = ef[i].X
                full_ot   = oh[i].X  # 0,1,2
                base_200  = 200 if y[i].X>0.5 else 0
                dbl_200   = 200 if w[i].X>0.5 else 0
                ot_pay    = 37.5 * full_ot
                floor_p   = 75 * extra_flo
                cost_i    = base_200 + dbl_200 + ot_pay + floor_p

                print(f"Attendant {i}:")
                print(f"  Rooms: {assigned_rooms}")
                print(f"  Total time = {tot_time:.2f} hr, area = {tot_area:.0f} sq ft")
                print(f"  Floors used = {flo_count:.0f}, extra floors = {extra_flo:.0f}")
                print(f"  OT hours (whole) = {int(full_ot)} => ${ot_pay:.2f}")
                print(f"  Cost breakdown => day={base_200}, double={dbl_200}, OT={ot_pay:.2f}, floors={floor_p:.2f}")
                print(f"  => Attendant cost = {cost_i:.2f}\n")
    else:
        print(f"No optimal solution found. Status = {m.status}")


if __name__ == "__main__":
    solve_covering_model_whole_OT("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/hotels.csv")