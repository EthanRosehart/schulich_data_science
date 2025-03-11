import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_welders_with_synergy():
    """
    Part (h):
      - We have an AI tool that provides base speed v[ii] for each welder i
        plus synergy v[ij] if both i and j are on the team.
      - We want to maximize sum of (v[ii]*x_i + sum_{i<j}(v[ij]*x_i*x_j))
      - Subject to the same constraints from Q2 (16 welders, min coverage, etc.).
    """

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------

    # We'll read the same "welders_data.csv" as before for each welder's 
    # proficiency, safety rating, etc. We'll interpret "Speed_Rating" 
    # as the "base speed" v[ii]. 
    # If you prefer to store base speed in a separate column, just adapt the code below.

    url_data = (
        "https://raw.githubusercontent.com/"
        "EthanRosehart/schulich_data_science/"
        "refs/heads/main/term3/Mid-term/welders_data.csv"
    )
    welders_df = pd.read_csv(url_data)
    n = len(welders_df)  # should be 144

    # We'll interpret "Speed_Rating" as base speed for each welder i:
    base_speed = welders_df["Speed_Rating"].tolist()

    # Safety, proficiency, experience, etc.:
    safety   = welders_df["Safety_Rating"].tolist()
    smaw     = welders_df["SMAW_Proficient"].tolist()
    gmaw     = welders_df["GMAW_Proficient"].tolist()
    fcaw     = welders_df["FCAW_Proficient"].tolist()
    gtaw     = welders_df["GTAW_Proficient"].tolist()
    exper    = welders_df["Experience_10_Years"].tolist()  # 0/1
    welder_id= welders_df["Welder_ID"].tolist()            # e.g. 1..144

    # We'll also read "welders_speed_data.csv" for synergy values v[ij].
    # We'll interpret it as a 144 x 144 matrix, where row i / col j => synergy of i&j
    url_synergy = (
        "https://raw.githubusercontent.com/"
        "EthanRosehart/schulich_data_science/"
        "refs/heads/main/term3/Mid-term/welders_speed_data.csv"
    )
    synergy_df = pd.read_csv(url_synergy, index_col=0)
    # synergy_df should be 144x144. We assume synergy_df.iloc[i,j] => v[ij].
    # If the diagonal synergy_df[i,i] includes the base speed, you'd skip the base_speed array 
    # or handle it carefully. For now we keep them separate.

    # Build sets for constraints
    # e.g., set_smaw, set_gmaw, set_fcaw, set_gtaw, set_experience, etc.
    # same as before
    set_smaw = [i for i in range(n) if smaw[i] == 1]
    set_gmaw = [i for i in range(n) if gmaw[i] == 1]
    set_fcaw = [i for i in range(n) if fcaw[i] == 1]
    set_gtaw = [i for i in range(n) if gtaw[i] == 1]
    set_smaw_gmaw = [i for i in range(n) if smaw[i] == 1 and gmaw[i] == 1]
    set_experience = [i for i in range(n) if exper[i] == 1]

    # speed=1 & safety=3 or 4
    set_speed1_safety34 = [
        i for i in range(n)
        if abs(base_speed[i] - 1.0) < 1e-9 and safety[i] in [3, 4]
    ]

    # For the "range" constraint: 
    set_70_100 = [i for i in range(n) if 70 <= welder_id[i] <= 100]
    set_101_130= [i for i in range(n) if 101 <= welder_id[i] <= 130]

    # ------------------------------------------------------------------
    # 2. Create Model
    # ------------------------------------------------------------------
    model = gp.Model("Welders_PartH")

    # x[i] = 1 if welder i is hired, else 0
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # ------------------------------------------------------------------
    # 3. Quadratic Objective:
    #    sum_i( base_speed[i]*x[i] ) + sum_{i<j}( synergy_df[i,j] * x[i]*x[j] )
    # ------------------------------------------------------------------
    obj_expr = gp.LinExpr()
    # add the linear part (base speeds)
    for i in range(n):
        obj_expr += base_speed[i] * x[i]

    # add the synergy part
    # (we do i<j to avoid double-counting if synergy[i,j] = synergy[j,i])
    for i in range(n):
        for j in range(i+1, n):
            v_ij = synergy_df.iloc[i, j]  # synergy between i & j
            # we add v_ij * x[i]*x[j]
            # We'll interpret synergy as additive. If your synergy is symmetrical,
            # synergy_df[i,j] = synergy_df[j,i].
            # If you only stored synergy above diagonal, that works too.
            obj_expr += v_ij * x[i]*x[j]

    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # ------------------------------------------------------------------
    # 4. Constraints (same as Q2):
    # ------------------------------------------------------------------

    # (a) exactly 16 welders
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 16, "HireExactly16")

    # (b) at least 50% proficient in BOTH => >=8
    model.addConstr(gp.quicksum(x[i] for i in set_smaw_gmaw) >= 8, "SMAW_GMAW_50pct")

    # (c) Each technique >=2
    model.addConstr(gp.quicksum(x[i] for i in set_smaw) >= 2, "AtLeast2_SMAW")
    model.addConstr(gp.quicksum(x[i] for i in set_gmaw) >= 2, "AtLeast2_GMAW")
    model.addConstr(gp.quicksum(x[i] for i in set_fcaw) >= 2, "AtLeast2_FCAW")
    model.addConstr(gp.quicksum(x[i] for i in set_gtaw) >= 2, "AtLeast2_GTAW")

    # (d) at least 30% with >10 yrs => 5
    model.addConstr(gp.quicksum(x[i] for i in set_experience) >= 5, "30pct_Experience")

    # (e) sum of safety >= 62.4 => average safety >=3.9
    model.addConstr(gp.quicksum(safety[i]*x[i] for i in range(n)) >= 62.4, "SafetyMinAvg")

    # (f) sum of speed >= 49.6 => average speed >=3.1
    #   NOTE: This is now overshadowed by the synergy objective but let's keep it
    model.addConstr(gp.quicksum(base_speed[i]*x[i] for i in range(n)) >= 49.6, "SpeedMinAvg")

    # (g) at least 3 with speed=1.0 & safety=3 or4
    model.addConstr(gp.quicksum(x[i] for i in set_speed1_safety34) >= 3, "AtLeast3_slowButSafe")

    # (h) [70..100] >= 1 +2*[101..130]
    model.addConstr(
        gp.quicksum(x[i] for i in set_70_100) >= 1 + 2*gp.quicksum(x[i] for i in set_101_130),
        "RangeConstraint"
    )

    # ------------------------------------------------------------------
    # 5. Solve as a Mixed-Integer Quadratic Program
    # ------------------------------------------------------------------
    # For a QP, you might want to set model.Params.NonConvex=2 if synergy can be negative or
    # if the matrix is indefinite. Usually synergy is symmetrical and can be handled. 
    # 
    # model.Params.NonConvex = 2

    model.optimize()

    # ------------------------------------------------------------------
    # 6. Print solution
    # ------------------------------------------------------------------
    if model.status == GRB.OPTIMAL:
        print(f"\n**** Part (h) Quadratic synergy model ****")
        print(f"Optimal objective = {model.objVal:.4f}")
        chosen = [i for i in range(n) if x[i].X > 0.5]
        print("Chosen welders (index -> base speed -> synergy indices if you like):")
        for i in chosen:
            print(f"  i={i}, base_speed={base_speed[i]}, safety={safety[i]}")
    else:
        print("No optimal solution or solver error. Status:", model.status)

if __name__=="__main__":
    solve_welders_with_synergy()
