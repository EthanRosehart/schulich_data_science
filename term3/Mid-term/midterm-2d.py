import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_welders_partd_no_speed1safety_constraint():
    # ----------------------------
    # 1. Read CSV data
    # ----------------------------
    url_welders = (
        "https://raw.githubusercontent.com/"
        "EthanRosehart/schulich_data_science/"
        "refs/heads/main/term3/Mid-term/welders_data.csv"
    )
    welders_df = pd.read_csv(url_welders)
    n_welders = len(welders_df)  # Expect 144

    # Extract columns
    welder_ids = welders_df["Welder_ID"].to_list()
    safety = welders_df["Safety_Rating"].to_list()
    speed = welders_df["Speed_Rating"].to_list()
    smaw = welders_df["SMAW_Proficient"].to_list()
    gmaw = welders_df["GMAW_Proficient"].to_list()
    fcaw = welders_df["FCAW_Proficient"].to_list()
    gtaw = welders_df["GTAW_Proficient"].to_list()
    experience = welders_df["Experience_10_Years"].to_list()  # 0/1

    # Build index sets for constraints
    set_smaw = [i for i in range(n_welders) if smaw[i] == 1]
    set_gmaw = [i for i in range(n_welders) if gmaw[i] == 1]
    set_fcaw = [i for i in range(n_welders) if fcaw[i] == 1]
    set_gtaw = [i for i in range(n_welders) if gtaw[i] == 1]
    set_smaw_gmaw = [i for i in range(n_welders) if smaw[i] == 1 and gmaw[i] == 1]
    set_experience = [i for i in range(n_welders) if experience[i] == 1]

    # For the constraint about [70..100] >= 1 + 2*[101..130]:
    set_70_100 = [i for i in range(n_welders) if 70 <= welder_ids[i] <= 100]
    set_101_130 = [i for i in range(n_welders) if 101 <= welder_ids[i] <= 130]

    # ----------------------------
    # 2. Create Model
    # ----------------------------
    model = gp.Model("Welders_Q2_PartD")

    # Binary decision vars: x[i] = 1 if welder i is hired
    x = model.addVars(n_welders, vtype=GRB.BINARY, name="x")

    # Objective: maximize total speed = sum of speed[i]*x[i]
    model.setObjective(gp.quicksum(speed[i]*x[i] for i in range(n_welders)), GRB.MAXIMIZE)

    # ----------------------------
    # 3. Constraints
    # ----------------------------

    # (a) Exactly 16 welders
    model.addConstr(gp.quicksum(x[i] for i in range(n_welders)) == 16, name="HireExactly16")

    # (b) At least 50% proficient in both SMAW & GMAW => >=8 of them
    model.addConstr(gp.quicksum(x[i] for i in set_smaw_gmaw) >= 8, name="SMAW_GMAW_50pct")

    # (c) For each technique, at least 2 welders
    model.addConstr(gp.quicksum(x[i] for i in set_smaw) >= 2, name="AtLeast2_SMAW")
    model.addConstr(gp.quicksum(x[i] for i in set_gmaw) >= 2, name="AtLeast2_GMAW")
    model.addConstr(gp.quicksum(x[i] for i in set_fcaw) >= 2, name="AtLeast2_FCAW")
    model.addConstr(gp.quicksum(x[i] for i in set_gtaw) >= 2, name="AtLeast2_GTAW")

    # (d) At least 30% with >10 years => sum x[i] in set_experience >= 4.8 => 5
    model.addConstr(gp.quicksum(x[i] for i in set_experience) >= 5, name="30pct_Experience")

    # (e) Average safety >=3.9 => sum_i(safety[i]*x[i]) >=3.9*16 => 62.4
    model.addConstr(gp.quicksum(safety[i]*x[i] for i in range(n_welders)) >= 62.4,
                    name="SafetyMinAvg")

    # (f) Average speed >=3.1 => sum_i(speed[i]*x[i]) >=3.1*16 => 49.6
    model.addConstr(gp.quicksum(speed[i]*x[i] for i in range(n_welders)) >= 49.6,
                    name="SpeedMinAvg")

    # (g) [70..100] >= 1 + 2*[101..130]
    model.addConstr(
        gp.quicksum(x[i] for i in set_70_100) >= 1 + 2*gp.quicksum(x[i] for i in set_101_130),
        name="RangeConstraint"
    )

    # NOTE: The constraint about "at least 3 welders with speed=1.0 and safety=3 or 4" is REMOVED here.

    # ----------------------------
    # 4. Optimize & Print
    # ----------------------------
    model.optimize()

    if model.status == GRB.OPTIMAL:
        obj_val = model.objVal
        print(f"\n**** Part (d) Model: No Speed=1 Safety=3or4 Constraint ****")
        print(f"Optimal objective (sum of speed) = {obj_val:.4f}")
        chosen = [i for i in range(n_welders) if x[i].X > 0.5]
        print("Chosen welders (index, ID, speed, safety):")
        for i in chosen:
            print(f"  i={i}, ID={welder_ids[i]}, speed={speed[i]}, safety={safety[i]}")
    else:
        print(f"Model not optimal. Status code: {model.status}")


if __name__ == "__main__":
    solve_welders_partd_no_speed1safety_constraint()
