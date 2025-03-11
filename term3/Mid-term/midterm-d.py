import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_ecoclean_blending_with_sensitivity():
    # ------------------------------
    # 1. Read CSV data from GitHub
    # ------------------------------
    url_materials = (
        "https://raw.githubusercontent.com/"
        "EthanRosehart/schulich_data_science/"
        "refs/heads/main/term3/Mid-term/blending_materials.csv"
    )
    url_blends = (
        "https://raw.githubusercontent.com/"
        "EthanRosehart/schulich_data_science/"
        "refs/heads/main/term3/Mid-term/blending_blends.csv"
    )

    materials_df = pd.read_csv(url_materials)
    blends_df = pd.read_csv(url_blends)

    num_materials = len(materials_df)  # 100
    num_blends = len(blends_df)        # 50

    availability = materials_df["availability"].to_list()
    cost = materials_df["cost"].to_list()
    p_max = materials_df["p_max"].to_list()

    demand = blends_df["demand"].to_list()
    quality_min = blends_df["quality_min"].to_list()
    quality_max = blends_df["quality_max"].to_list()

    # Build the 2D list for quality[i][j]
    quality = []
    for i in range(num_blends):
        row_quality = []
        for j in range(1, num_materials + 1):
            col_name = f"quality_{j}"
            row_quality.append(blends_df.loc[i, col_name])
        quality.append(row_quality)

    # -------------------------------------------
    # 2. Create Gurobi Model and Decision Vars
    # -------------------------------------------
    model = gp.Model("EcoCleanBlending_Sensitivity")

    # Decision variables x[i,j]
    x = model.addVars(num_blends, num_materials, lb=0,
                      vtype=GRB.CONTINUOUS, name="x")

    # -------------------------------------------
    # 3. Objective: Minimize total production cost
    # -------------------------------------------
    model.setObjective(
        gp.quicksum(cost[j] * x[i, j]
                    for i in range(num_blends)
                    for j in range(num_materials)),
        GRB.MINIMIZE
    )

    # -------------------------------------------
    # 4. Constraints
    # -------------------------------------------
    # Demand constraints
    for i in range(num_blends):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(num_materials)) == demand[i],
            name=f"demand_of_blend_{i}"
        )

    # Availability constraints
    for j in range(num_materials):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_blends)) <= availability[j],
            name=f"availability_of_mat_{j}"
        )

    # Quality constraints (average quality)
    for i in range(num_blends):
        lhs_expr = gp.quicksum(quality[i][j] * x[i, j]
                               for j in range(num_materials))

        # Minimum quality
        c_min = model.addConstr(lhs_expr >= quality_min[i] * demand[i],
                                name=f"quality_min_blend_{i}")

        # Maximum quality
        c_max = model.addConstr(lhs_expr <= quality_max[i] * demand[i],
                                name=f"quality_max_blend_{i}")

    # Proportion constraints
    for i in range(num_blends):
        for j in range(num_materials):
            model.addConstr(
                x[i, j] <= p_max[j] * demand[i],
                name=f"pmax_blend_{i}_mat_{j}"
            )

    # -------------------------------------------
    # 5. Optimize
    # -------------------------------------------
    model.optimize()

    # -------------------------------------------
    # 6. Output & Sensitivity
    # -------------------------------------------
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal Cost = {model.objVal:,.2f}\n")

        # (A) Print non-zero usage
        for i in range(num_blends):
            for j in range(num_materials):
                val = x[i, j].X
                if val > 1e-6:
                    print(f"Blend {i+1}, Material {j+1}: {val:.3f} tons")

        print("\n--- SENSITIVITY ANALYSIS ---\n")

        # (B) For each blend i, retrieve shadow price (Pi) and slack on min-quality constraint
        for i in range(num_blends):
            c_min = model.getConstrByName(f"quality_min_blend_{i}")
            pi_min = c_min.Pi
            slack_min = c_min.Slack
            print(f"Blend {i+1} (Qmin): Pi={pi_min:.4f}, Slack={slack_min:.4f}")

        # You could also retrieve Pi & Slack for max-quality constraints, availability, etc.
        # c_max = model.getConstrByName(f"quality_max_blend_{i}"), etc.

    else:
        print(f"Model not optimal. Status: {model.status}")

if __name__ == "__main__":
    solve_ecoclean_blending_with_sensitivity()