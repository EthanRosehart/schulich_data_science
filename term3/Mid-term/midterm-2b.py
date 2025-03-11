import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_welder_models():
    # ------------------------------------------------------------
    # 1. Read welder data
    # ------------------------------------------------------------
    url_welders = (
        "https://raw.githubusercontent.com/"
        "EthanRosehart/schulich_data_science/"
        "refs/heads/main/term3/Mid-term/welders_data.csv"
    )

    welders_df = pd.read_csv(url_welders)

    # Let's store some convenient arrays / sets:
    # We'll use zero-based indexing in Python, so each row is welder i in [0..143].
    n_welders = len(welders_df)  # Should be 144

    # Each welder has:
    # "Welder_ID,Safety_Rating,Speed_Rating,SMAW_Proficient,GMAW_Proficient,FCAW_Proficient,GTAW_Proficient,Experience_10_Years"

    # We'll convert them to lists or arrays for fast access:
    safety = welders_df["Safety_Rating"].to_list()
    speed = welders_df["Speed_Rating"].to_list()
    smaw = welders_df["SMAW_Proficient"].to_list()   # 0/1
    gmaw = welders_df["GMAW_Proficient"].to_list()   # 0/1
    fcaw = welders_df["FCAW_Proficient"].to_list()   # 0/1
    gtaw = welders_df["GTAW_Proficient"].to_list()   # 0/1
    experience = welders_df["Experience_10_Years"].to_list()  # 0/1

    # The question says we want to hire exactly 16 welders,
    # so sum x_i = 16.

    # Additional constraints (based on typical reading of Q2):
    # 1) At least 50% must be proficient in BOTH SMAW & GMAW
    #    => sum_i x_i*(SMAW & GMAW) >= 0.5 * 16 => >= 8
    # 2) For each technique, at least 2 welders are proficient
    # 3) At least 30% have >10 yrs => sum_i x_i*(experience=1) >= 0.3*16 => >=4.8 => 5
    # 4) Average rating of safety >= 3.9 => sum_i(safety[i]*x_i)/16 >= 3.9 => sum_i(...)>=62.4
    # 5) Average rating of speed >= 3.1 => sum_i(speed[i]*x_i)/16 >=3.1 => sum_i(...)>=49.6
    # 6) At least 3 welders have speed=1.0 and safety=3 or 4
    # 7) # [70..100] must be at least 1 + 2*[101..130] (the question mentions something like that)
    #
    # We'll incorporate these. If your question doesn't need them all, feel free to remove them.

    # Let's define sets / helpers for constraints:
    #  - set of welders proficient in each technique
    #  - set of welders that are proficient in BOTH SMAW & GMAW
    #  - set of welders with speed=1 and safety in {3,4} if that constraint is relevant
    #  - sets for range 70..100, 101..130, etc. We'll note that Welder_ID might be 1-based in the CSV. We'll check.

    # We must see how "Welder_ID" aligns with row index. Possibly welder i => ID i+1. We'll do:
    welder_ids = welders_df["Welder_ID"].to_list()  # Might be 1..144

    # We'll build small "index sets" for each constraint:
    set_smaw = [i for i in range(n_welders) if smaw[i] == 1]
    set_gmaw = [i for i in range(n_welders) if gmaw[i] == 1]
    set_fcaw = [i for i in range(n_welders) if fcaw[i] == 1]
    set_gtaw = [i for i in range(n_welders) if gtaw[i] == 1]
    set_smaw_gmaw = [i for i in range(n_welders) if smaw[i] == 1 and gmaw[i] == 1]
    set_experience = [i for i in range(n_welders) if experience[i] == 1]

    # speed=1 and safety=3 or 4:
    set_speed1_safety34 = [i for i in range(n_welders)
                           if abs(speed[i] - 1.0) < 1e-9 and (safety[i] == 3 or safety[i] == 4)]

    # We interpret "range 70..100" and "101..130" as referencing the *Welder_ID*, not row index:
    set_70_100 = [i for i in range(n_welders) if 70 <= welder_ids[i] <= 100]
    set_101_130 = [i for i in range(n_welders) if 101 <= welder_ids[i] <= 130]

    # Summaries for convenience:
    # print("Proficient in SMAW & GMAW:", set_smaw_gmaw)
    # etc.

    # We'll define a helper function to build & solve a model
    def build_solve_model(vtype_for_x):
        """
        vtype_for_x = GRB.BINARY or GRB.CONTINUOUS
        Returns (model, x) for further inspection
        """
        model = gp.Model("WeldersSelection")

        # Decision vars: x[i] in {0,1} or [0,1]
        x = model.addVars(n_welders, lb=0, ub=1, vtype=vtype_for_x, name="x")

        # Objective: maximize sum of speed[i]*x[i]
        # (equivalent to maximizing average speed since # of hires=16 is fixed)
        model.setObjective(gp.quicksum(speed[i]*x[i] for i in range(n_welders)), GRB.MAXIMIZE)

        # 1) sum_i x_i = 16
        model.addConstr(gp.quicksum(x[i] for i in range(n_welders)) == 16, name="HireExactly16")

        # 2) At least half proficient in BOTH => sum_i x_i in set_smaw_gmaw >= 8
        model.addConstr(gp.quicksum(x[i] for i in set_smaw_gmaw) >= 8, name="SMAW_GMAW_50pct")

        # 3) Each technique >=2
        model.addConstr(gp.quicksum(x[i] for i in set_smaw) >= 2, name="AtLeast2_SMAW")
        model.addConstr(gp.quicksum(x[i] for i in set_gmaw) >= 2, name="AtLeast2_GMAW")
        model.addConstr(gp.quicksum(x[i] for i in set_fcaw) >= 2, name="AtLeast2_FCAW")
        model.addConstr(gp.quicksum(x[i] for i in set_gtaw) >= 2, name="AtLeast2_GTAW")

        # 4) At least 30% >10 yrs => sum x_i in set_experience >= 0.3*16 => >=4.8 => so >=5
        model.addConstr(gp.quicksum(x[i] for i in set_experience) >= 5, name="30pct_Experience")

        # 5) Average safety >=3.9 => sum(safety[i]*x[i]) >= 3.9*16=62.4
        model.addConstr(gp.quicksum(safety[i]*x[i] for i in range(n_welders)) >= 62.4,
                        name="SafetyMinAvg")

        # 6) Average speed >=3.1 => sum(speed[i]*x[i]) >=3.1*16=49.6
        #    Actually we are maximizing sum of speed, so we can just keep it a constraint:
        model.addConstr(gp.quicksum(speed[i]*x[i] for i in range(n_welders)) >= 49.6,
                        name="SpeedMinAvg")

        # 7) At least 3 welders with speed=1.0 and safety=3 or 4
        model.addConstr(gp.quicksum(x[i] for i in set_speed1_safety34) >= 3,
                        name="AtLeast3_slowButSafe")

        # 8) # from [70..100] >= 1 + 2 * # from [101..130]
        #    sum x in 70..100 >= 1 + 2 sum x in 101..130
        model.addConstr(gp.quicksum(x[i] for i in set_70_100)
                        >= 1 + 2*gp.quicksum(x[i] for i in set_101_130),
                        name="RangeConstraint")

        # Solve
        model.optimize()
        return model, x

    # ------------------------------------------------------------
    # 2. Build & Solve the Binary Model
    # ------------------------------------------------------------
    model_binary, x_binary = build_solve_model(GRB.BINARY)
    if model_binary.status == GRB.OPTIMAL:
        obj_binary = model_binary.objVal
        # Grab the chosen welders
        chosen_bin = [i for i in range(n_welders) if x_binary[i].X > 0.5]
        print("\n=== BINARY MODEL ===")
        print(f"Objective (sum of speeds) = {obj_binary:.4f}")
        print("Chosen welders (index, ID, speed, safety):")
        for i in chosen_bin:
            print(f"  i={i}, ID={welder_ids[i]}, speed={speed[i]}, safety={safety[i]}")
    else:
        print("Binary model not optimal. Status:", model_binary.status)

    # ------------------------------------------------------------
    # 3. Build & Solve the Linear Relaxation
    # ------------------------------------------------------------
    model_relax, x_relax = build_solve_model(GRB.CONTINUOUS)
    if model_relax.status == GRB.OPTIMAL:
        obj_relax = model_relax.objVal
        print("\n=== RELAXATION MODEL ===")
        print(f"Objective (sum of speeds) = {obj_relax:.4f}")
        # If you want to see fractional x_i:
        # We'll just show any x_i > 1e-6
        print("Non-zero x_i (welder index, fraction, speed, safety):")
        for i in range(n_welders):
            val = x_relax[i].X
            if val > 1e-6:
                print(f"  i={i}, fraction={val:.3f}, ID={welder_ids[i]},"
                      f" speed={speed[i]}, safety={safety[i]}")
    else:
        print("Relaxation model not optimal. Status:", model_relax.status)

if __name__ == "__main__":
    solve_welder_models()
