import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_ecoclean_blending_shadow_73():
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
        for j in range(1, num_materials+1):
            col_name = f"quality_{j}"
            row_quality.append(blends_df.loc[i, col_name])
        quality.append(row_quality)

    # -------------------------------------------
    # 2. Create Gurobi Model and Decision Vars
    # -------------------------------------------
    model = gp.Model("EcoCleanBlending_Shadow73")

    # x[i,j] = tons of raw material j used in blend i
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

    # Quality (min & max)
    for i in range(num_blends):
        lhs_expr = gp.quicksum(quality[i][j] * x[i, j]
                               for j in range(num_materials))

        # Min quality
        model.addConstr(lhs_expr >= quality_min[i] * demand[i],
                        name=f"quality_min_blend_{i}")
        # Max quality
        model.addConstr(lhs_expr <= quality_max[i] * demand[i],
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
    # 6. Output Results
    # -------------------------------------------
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal production cost: {model.objVal:,.2f}")

        # Report usage for non-zero x[i,j]
        for i in range(num_blends):
            for j in range(num_materials):
                val = x[i, j].X
                if val > 1e-6:
                    print(f"Blend {i+1}, Material {j+1}: {val:.3f} tons")

        # -------------------------------------------
        # SHADOW PRICE for raw material #73 (index=72)
        # -------------------------------------------
        constr_name = f"availability_of_mat_72"
        avail_constr_73 = model.getConstrByName(constr_name)

        if avail_constr_73 is None:
            print(f"\nConstraint {constr_name} not found!")
        else:
            pi_73 = avail_constr_73.Pi
            slack_73 = avail_constr_73.Slack
            print(f"\n--- Shadow Price for availability of Material #73 ---")
            print(f"Constraint Name: {constr_name}")
            print(f"  Pi (dual value)    = {pi_73:.4f}")
            print(f"  Slack              = {slack_73:.4f}")
            # Interpretation for a cost-minimization with <= constraint:
            #   - If Pi < 0 => each additional ton helps reduce cost by |Pi|.
            #   - Slack=0 means we're at capacity for #73.
        print()
    else:
        print(f"No optimal solution found. Status code: {model.status}")

if __name__ == "__main__":
    solve_ecoclean_blending_shadow_73()
