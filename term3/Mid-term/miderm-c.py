import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_ecoclean_blending_rc_j5():
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
    model = gp.Model("EcoCleanBlending_RC_j5")

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
    # Demand
    for i in range(num_blends):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(num_materials)) == demand[i],
            name=f"demand_of_blend_{i}"
        )

    # Availability
    for j in range(num_materials):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_blends)) <= availability[j],
            name=f"availability_of_mat_{j}"
        )

    # Quality
    for i in range(num_blends):
        lhs_expr = gp.quicksum(quality[i][j] * x[i, j]
                               for j in range(num_materials))
        model.addConstr(lhs_expr >= quality_min[i] * demand[i],
                        name=f"quality_min_blend_{i}")
        model.addConstr(lhs_expr <= quality_max[i] * demand[i],
                        name=f"quality_max_blend_{i}")

    # Proportion
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

        # Check usage for raw material j=4 (i.e., material #5)
        # We'll note if it's used in any blend
        used_in_any_blend = False
        for i in range(num_blends):
            val = x[i, 4].X  # j=4 -> material #5
            if val > 1e-6:
                used_in_any_blend = True
                print(f"Blend {i+1}, Material 5 usage: {val:.3f} tons")

        if not used_in_any_blend:
            print("Material 5 is not used in any blend at the current cost.")

        # 7. Reduced Cost for j=4 (Material #5) if not used
        print("\nReduced Costs for j=4 (Material #5) in each blend:")
        for i in range(num_blends):
            # Because we minimize, a positive RC indicates how much the cost
            # must drop before x[i,4] could become positive.
            rc = x[i, 4].RC
            val = x[i, 4].X
            print(f"  Blend {i+1}: usage={val:.6f}, RC={rc:.4f}")

    else:
        print(f"No optimal solution found. Status code: {model.status}")


if __name__ == "__main__":
    solve_ecoclean_blending_rc_j5()
