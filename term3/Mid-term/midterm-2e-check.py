import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def model_without_7_constraints():
    # ----------------------------
    # 1) Read Data
    # ----------------------------
    url_welders = (
        "https://raw.githubusercontent.com/"
        "EthanRosehart/schulich_data_science/"
        "refs/heads/main/term3/Mid-term/welders_data.csv"
    )
    welders_df = pd.read_csv(url_welders)

    n_welders = len(welders_df)  # Expect 144
    welder_ids = welders_df["Welder_ID"].to_list()
    safety = welders_df["Safety_Rating"].to_list()
    speed = welders_df["Speed_Rating"].to_list()
    smaw = welders_df["SMAW_Proficient"].to_list()   # 0/1
    gmaw = welders_df["GMAW_Proficient"].to_list()   # 0/1
    experience = welders_df["Experience_10_Years"].to_list()  # 0/1
    fcaw = welders_df["FCAW_Proficient"].to_list()
    gtaw = welders_df["GTAW_Proficient"].to_list()

    # Build subsets
    set_smaw_gmaw = [i for i in range(n_welders) if smaw[i] == 1 and gmaw[i] == 1]

    # speed=1 & safety=3 or 4
    set_speed1_safety34 = [i for i in range(n_welders)
                           if abs(speed[i] - 1.0) < 1e-9 and safety[i] in (3,4)]

    # ----------------------------
    # 2) Create Model
    # ----------------------------
    model = gp.Model("ReducedModel_No7Constraints")
    x = model.addVars(n_welders, vtype=GRB.BINARY, name="x")

    # ----------------------------
    # 3) Objective: maximize sum of speeds
    # ----------------------------
    model.setObjective(gp.quicksum(speed[i]*x[i] for i in range(n_welders)), GRB.MAXIMIZE)

    # ----------------------------
    # 4) Keep only the four "definitely active" constraints
    # ----------------------------

    # (a) sum_i x_i = 16
    model.addConstr(gp.quicksum(x[i] for i in range(n_welders)) == 16,
                    name="HireExactly16")

    # (b) >= 8 welders proficient in BOTH SMAW & GMAW
    model.addConstr(gp.quicksum(x[i] for i in set_smaw_gmaw) >= 8,
                    name="SMAW_GMAW_50pct")

    # (c) sum of safety >= 62.4 => average safety >=3.9
    model.addConstr(gp.quicksum(safety[i]*x[i] for i in range(n_welders)) >= 62.4,
                    name="SafetyMinAvg")

    # (d) at least 3 welders with speed=1 & safety in {3,4}
    model.addConstr(gp.quicksum(x[i] for i in set_speed1_safety34) >= 3,
                    name="AtLeast3_slowButSafe")

    # ----------------------------
    # 5) Optimize & Print
    # ----------------------------
    model.optimize()
    if model.status == GRB.OPTIMAL:
        obj_val = model.objVal
        print("\n** Reduced Model (No 7 Possibly Redundant Constraints) **")
        print(f"Objective (sum of speeds) = {obj_val:.4f}")
        chosen = [i for i in range(n_welders) if x[i].X > 0.5]
        print("Chosen welders (index, speed, safety):")
        for i in chosen:
            print(f"  i={i}, speed={speed[i]}, safety={safety[i]}")
    else:
        print("Model not optimal or infeasible. Status:", model.status)

if __name__=="__main__":
    model_without_7_constraints()
