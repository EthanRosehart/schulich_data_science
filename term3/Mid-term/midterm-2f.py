import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def enumerate_optimal_solutions():
    # -----------------------------
    # 1) Read CSV data
    # -----------------------------
    url_welders = (
        "https://raw.githubusercontent.com/"
        "EthanRosehart/schulich_data_science/"
        "refs/heads/main/term3/Mid-term/welders_data.csv"
    )
    welders_df = pd.read_csv(url_welders)

    n_welders = len(welders_df)  # 144
    welder_ids = welders_df["Welder_ID"].to_list()
    safety = welders_df["Safety_Rating"].to_list()
    speed = welders_df["Speed_Rating"].to_list()
    smaw = welders_df["SMAW_Proficient"].to_list()
    gmaw = welders_df["GMAW_Proficient"].to_list()
    fcaw = welders_df["FCAW_Proficient"].to_list()
    gtaw = welders_df["GTAW_Proficient"].to_list()
    experience = welders_df["Experience_10_Years"].to_list()

    # Build sets for constraints
    set_smaw = [i for i in range(n_welders) if smaw[i] == 1]
    set_gmaw = [i for i in range(n_welders) if gmaw[i] == 1]
    set_smaw_gmaw = [i for i in range(n_welders) if smaw[i] == 1 and gmaw[i] == 1]

    set_fcaw = [i for i in range(n_welders) if fcaw[i] == 1]
    set_gtaw = [i for i in range(n_welders) if gtaw[i] == 1]
    set_experience = [i for i in range(n_welders) if experience[i] == 1]

    # speed=1 & safety in {3,4}
    set_speed1_safety34 = [i for i in range(n_welders)
                           if abs(speed[i] - 1.0)<1e-9 and safety[i] in (3,4)]
    # For the "range" constraint
    set_70_100 = [i for i in range(n_welders) if 70 <= welder_ids[i] <= 100]
    set_101_130 = [i for i in range(n_welders) if 101 <= welder_ids[i] <= 130]

    # -----------------------------
    # 2) Create the baseline model
    # -----------------------------
    model = gp.Model("Welders_EnumerateOptimal")
    x = model.addVars(n_welders, vtype=GRB.BINARY, name="x")

    # Objective: maximize sum of speeds
    model.setObjective(gp.quicksum(speed[i]*x[i] for i in range(n_welders)), GRB.MAXIMIZE)

    # Constraints: same as Q2 (b)
    model.addConstr(gp.quicksum(x[i] for i in range(n_welders)) == 16, name="HireExactly16")

    # At least 50% in SMAW & GMAW => >=8
    model.addConstr(gp.quicksum(x[i] for i in set_smaw_gmaw) >= 8, name="SMAW_GMAW_50pct")

    # Each technique >=2
    model.addConstr(gp.quicksum(x[i] for i in set_smaw) >= 2, name="AtLeast2_SMAW")
    model.addConstr(gp.quicksum(x[i] for i in set_gmaw) >= 2, name="AtLeast2_GMAW")
    model.addConstr(gp.quicksum(x[i] for i in set_fcaw) >= 2, name="AtLeast2_FCAW")
    model.addConstr(gp.quicksum(x[i] for i in set_gtaw) >= 2, name="AtLeast2_GTAW")

    # At least 30% with >10 yrs => >=5
    model.addConstr(gp.quicksum(x[i] for i in set_experience) >= 5, name="30pct_Experience")

    # sum of safety >=3.9 => >=62.4
    model.addConstr(gp.quicksum(safety[i]*x[i] for i in range(n_welders)) >= 62.4, 
                    name="SafetyMinAvg")

    # sum of speed >=3.1 => >=49.6
    model.addConstr(gp.quicksum(speed[i]*x[i] for i in range(n_welders)) >= 49.6,
                    name="SpeedMinAvg")

    # At least 3 with speed=1.0 & safety=3 or 4
    model.addConstr(gp.quicksum(x[i] for i in set_speed1_safety34) >= 3,
                    name="AtLeast3_slowButSafe")

    # Range constraint
    model.addConstr(
        gp.quicksum(x[i] for i in set_70_100) >= 1 + 2*gp.quicksum(x[i] for i in set_101_130),
        name="RangeConstraint"
    )

    # -----------------------------
    # 3) Setup the solution pool
    # -----------------------------
    # We'll do an exhaustive search if the problem is small enough or
    # it might be large. We'll attempt to get all solutions anyway.
    model.Params.PoolSearchMode = 2   # 2 = do a comprehensive search for solutions
    model.Params.PoolSolutions = 1000000  # up to 1e6 solutions stored, if feasible
    model.Params.PoolGap = 0.0        # gather all solutions of the same objective
    # Possibly also set model.Params.PoolIntCompositions= # advanced usage

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.INTERRUPTED:
        # get best objective
        best_obj = model.objVal
        # total number of solutions in the solution pool
        sol_count = model.SolCount

        print(f"\n** Solve finished: best objective = {best_obj:.4f}, found {sol_count} solutions in the pool. **")

        # We can iterate over each solution in the pool and see how many also have that same objective
        same_obj_count = 0
        max_show = min(sol_count, 10)   # we might just show first 10

        for sol_idx in range(sol_count):
            model.setParam(GRB.Param.SolutionNumber, sol_idx)
            obj_sol = model.PoolObjVal
            if abs(obj_sol - best_obj) < 1e-6:
                same_obj_count += 1

        print(f"Number of solutions with objective == {best_obj:.4f} = {same_obj_count}")

        # (Optional) show first few solutions
        # We'll do only if we want to see them
        # for sol_idx in range(max_show):
        #     model.setParam(GRB.Param.SolutionNumber, sol_idx)
        #     obj_sol = model.PoolObjVal
        #     # gather which welders are chosen
        #     chosen = []
        #     for i in range(n_welders):
        #         xval = x[i].Xn  # Xn is the var value in solutionNumber
        #         if xval > 0.5:
        #             chosen.append(i)
        #     print(f"--- Solution {sol_idx}, obj={obj_sol:.2f}, chosen={chosen}")
    else:
        print("No optimal solution or solver error. Status:", model.status)

if __name__ == "__main__":
    enumerate_optimal_solutions()
