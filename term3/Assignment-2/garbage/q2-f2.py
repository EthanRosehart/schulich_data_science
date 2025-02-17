import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_covering_model_integer(hotels_csv_path):
    """
    Builds and solves the original covering model with BINARY variables.
    Returns the (solved) integer model object m.
    """
    # --- 1) Read data ---
    df = pd.read_csv(hotels_csv_path)
    rooms = df["Room_ID"].tolist()
    floor_of = dict(zip(df["Room_ID"], df["Floor"]))
    area_of  = dict(zip(df["Room_ID"], df["Square_Feet"]))
    time_of  = dict(zip(df["Room_ID"], df["Cleaning_Time_Hours"]))
    floors_list = sorted(df["Floor"].unique())

    # Attendant set
    I = 8
    attendants = range(I)

    # --- 2) Create Model ---
    m = gp.Model("HotelCovering_Integer")

    # --- 3) Decision Variables (BINARY / INTEGER) ---
    x = {}
    for i in attendants:
        for r in rooms:
            x[(i,r)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{r}")

    y = {}
    for i in attendants:
        y[i] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    w = {}
    for i in attendants:
        w[i] = m.addVar(vtype=GRB.BINARY, name=f"w_{i}")

    fvar = {}
    for i in attendants:
        for fl in floors_list:
            fvar[(i,fl)] = m.addVar(vtype=GRB.BINARY, name=f"f_{i}_{fl}")

    # For partial OT example (from the original simpler approach)
    T  = {}
    OT = {}
    for i in attendants:
        T[i]  = m.addVar(vtype=GRB.CONTINUOUS, lb=0,  name=f"T_{i}")
        OT[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0,  name=f"OT_{i}")

    floors_i = {}
    for i in attendants:
        floors_i[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=4, name=f"floors_{i}")

    ef = {}
    for i in attendants:
        ef[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=2, name=f"ef_{i}")

    m.update()

    # --- 4) Constraints ---
    # Cover each room exactly once
    for r in rooms:
        m.addConstr(gp.quicksum(x[(i,r)] for i in attendants) == 1)

    # If i cleans any room => y[i]=1
    num_rooms = len(rooms)
    for i in attendants:
        m.addConstr(gp.quicksum(x[(i,r)] for r in rooms) <= num_rooms * y[i])

    # T[i] = sum of cleaning times
    for i in attendants:
        m.addConstr(T[i] == gp.quicksum(time_of[r]*x[(i,r)] for r in rooms))
        # At most 10 hr if used
        m.addConstr(T[i] <= 10*y[i])

    # OT >= T[i]-8, up to 2
    for i in attendants:
        m.addConstr(OT[i] >= T[i] - 8)
        m.addConstr(OT[i] <= 2*y[i])
        m.addConstr(OT[i] <= T[i])

    # Square footage => if sum(area)>3500 => w[i]
    total_area = df["Square_Feet"].sum()
    for i in attendants:
        area_cleaned = gp.quicksum(area_of[r]*x[(i,r)] for r in rooms)
        m.addConstr(area_cleaned <= 3500 + total_area*w[i])
        m.addConstr(3500*w[i] <= area_cleaned)

    # Floor usage
    for i in attendants:
        for fl in floors_list:
            rooms_on_floor = [r for r in rooms if floor_of[r]==fl]
            m.addConstr(gp.quicksum(x[(i,r)] for r in rooms_on_floor)
                        <= len(rooms_on_floor)*fvar[(i,fl)])
        m.addConstr(floors_i[i] == gp.quicksum(fvar[(i,fl)] for fl in floors_list))
        m.addConstr(floors_i[i] <= 4*y[i])
        m.addConstr(ef[i] >= floors_i[i]-2)
        m.addConstr(ef[i] <= floors_i[i])

    # --- 5) Objective ---
    # cost_i = 200*y[i] + 200*w[i] + 37.5*OT[i] + 75*ef[i]
    obj_expr = gp.LinExpr()
    for i in attendants:
        daypay  = 200*y[i]
        doublep = 200*w[i]
        otpay   = 37.5*OT[i]
        flpay   = 75*ef[i]
        obj_expr += (daypay + doublep + otpay + flpay)

    m.setObjective(obj_expr, GRB.MINIMIZE)

    # Solve the integer model
    m.optimize()
    return m

def solve_part_f_with_relax(hotels_csv_path):
    """
    Builds the original integer model, then calls model.relax() 
    to solve the LP relaxation. Prints that solution's objective 
    and any fractional variable values.
    """
    # Build and solve the original integer model first (not strictly necessary to solve it, 
    # but typically you'd do so to see the difference).
    m_int = solve_covering_model_integer(hotels_csv_path)
    if m_int.status == GRB.OPTIMAL:
        print(f"===> Original Integer Optimum: {m_int.objVal:.2f}")
    else:
        print(f"Integer model not optimal. Status={m_int.status}")

    # Now create the relaxed model
    relaxed = m_int.relax()  # This returns a new model with continuous versions of all Int/Bin vars
    relaxed.optimize()

    if relaxed.status == GRB.OPTIMAL:
        print(f"\n===> LP Relaxation objective: {relaxed.objVal:.2f}")
        # Optionally, we can print variable values (which may be fractional)
        for v in relaxed.getVars():
            if abs(v.X) > 1e-6:
                print(f"  {v.VarName} = {v.X:.4f}")
    else:
        print(f"Relaxed model not optimal. Status={relaxed.status}")
if __name__=="__main__":
    solve_part_f_with_relax("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/hotels.csv")