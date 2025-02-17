import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_covering_model_relaxed_manual(hotels_csv_path):
    """
    Builds a *linear relaxation* of the original covering model 
    by setting variables that were BINARY/INT to 
    CONTINUOUS with appropriate bounds (0..1 or 0..2, etc.).
    Then solves that model. 
    """
    df = pd.read_csv(hotels_csv_path)
    rooms = df["Room_ID"].tolist()
    floor_of = dict(zip(df["Room_ID"], df["Floor"]))
    area_of  = dict(zip(df["Room_ID"], df["Square_Feet"]))
    time_of  = dict(zip(df["Room_ID"], df["Cleaning_Time_Hours"]))
    floors_list = sorted(df["Floor"].unique())

    I = 8
    attendants = range(I)

    m = gp.Model("HotelCovering_ManualRelax")

    # Decision vars now continuous:
    # x[i,r] in [0,1] (instead of BINARY)
    x = {}
    for i in attendants:
        for r in rooms:
            x[(i,r)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"x_{i}_{r}")

    # y[i] in [0,1] (instead of BINARY)
    y = {}
    for i in attendants:
        y[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"y_{i}")

    # w[i] in [0,1]
    w = {}
    for i in attendants:
        w[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"w_{i}")

    # fvar[i,k] in [0,1]
    fvar = {}
    for i in attendants:
        for fl in floors_list:
            fvar[(i,fl)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"f_{i}_{fl}")

    # T[i], OT[i] remain continuous
    T  = {}
    OT = {}
    for i in attendants:
        T[i]  = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"T_{i}")
        OT[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"OT_{i}")

    # floors_i[i] in [0,4] (instead of integer up to 4)
    floors_i = {}
    for i in attendants:
        floors_i[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=4, name=f"floors_{i}")

    # ef[i] in [0,2] (instead of integer)
    ef = {}
    for i in attendants:
        ef[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2, name=f"ef_{i}")

    m.update()

    # Constraints are the same except the variables can now be fractional:
    for r in rooms:
        m.addConstr(gp.quicksum(x[(i,r)] for i in attendants) == 1)

    num_rooms = len(rooms)
    for i in attendants:
        m.addConstr(
            gp.quicksum(x[(i,r)] for r in rooms) <= num_rooms * y[i]
        )
        m.addConstr(T[i] == gp.quicksum(time_of[r]*x[(i,r)] for r in rooms))
        m.addConstr(T[i] <= 10*y[i])
        m.addConstr(OT[i] >= T[i] - 8)
        m.addConstr(OT[i] <= 2*y[i])
        m.addConstr(OT[i] <= T[i])

        cleaned_area = gp.quicksum(area_of[r]*x[(i,r)] for r in rooms)
        total_area = df["Square_Feet"].sum()
        m.addConstr(cleaned_area <= 3500 + total_area*w[i])
        m.addConstr(3500*w[i] <= cleaned_area)

        for fl in floors_list:
            rooms_on_floor = [r for r in rooms if floor_of[r]==fl]
            m.addConstr(
                gp.quicksum(x[(i,r)] for r in rooms_on_floor)
                <= len(rooms_on_floor)*fvar[(i,fl)]
            )

        m.addConstr(
            floors_i[i] == gp.quicksum(fvar[(i,fl)] for fl in floors_list)
        )
        m.addConstr(floors_i[i] <= 4*y[i])
        m.addConstr(ef[i] >= floors_i[i]-2)
        m.addConstr(ef[i] <= floors_i[i])

    # Objective is the same expression
    obj_expr = gp.LinExpr()
    for i in attendants:
        daypay  = 200*y[i]
        doublep = 200*w[i]
        otpay   = 37.5*OT[i]
        flpay   = 75*ef[i]
        obj_expr += (daypay + doublep + otpay + flpay)

    m.setObjective(obj_expr, GRB.MINIMIZE)

    # Solve the *relaxed* model
    m.optimize()

    # Print results
    if m.status == GRB.OPTIMAL:
        print(f"Optimal solution (Relaxed Manual) = {m.objVal:.2f}\n")
        # Possibly show fractional solutions
        for v in m.getVars():
            if abs(v.X) > 1e-6 and v.VarName.startswith("x_"):
                print(f"{v.VarName} = {v.X:.4f}")
    else:
        print(f"Model not optimal. Status={m.status}")

if __name__ == "__main__":
    solve_covering_model_relaxed_manual("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/hotels.csv")