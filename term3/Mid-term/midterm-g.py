import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_ecoclean_blending_squares_objective():
    """
    Part (g):
      Keep all the linear constraints from the original problem,
      but the objective is now sum_{i,j}( cost_j * x[i,j]^2 ).
    """
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
    # 2. Create Gurobi Model
    # -------------------------------------------
    model = gp.Model("EcoCleanBlending_SumOfSquaresObj")

    # If Gurobi flags it as non-convex for some reason (shouldn't if cost >= 0),
    # we might need:
    # model.Params.NonConvex = 2

    # -------------------------------------------
    # 3. Decision Variables
    # -------------------------------------------
    # x[i,j] = tons of raw material j used in blend i
    x = model.addVars(num_blends, num_materials, lb=0,
                      vtype=GRB.CONTINUOUS, name="x")

    # -------------------------------------------
    # 4. Quadratic Objective: sum_{i,j}( cost_j * x[i,j]^2 )
    # -------------------------------------------
    obj_expr = gp.quicksum(cost[j] * (x[i, j]*x[i, j])
                           for i in range(num_blends)
                           for j in range(num_materials))
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # -------------------------------------------
    # 5. Constraints (same linear constraints as original)
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

    # Quality constraints
    for i in range(num_blends):
        lhs_expr = gp.quicksum(quality[i][j] * x[i, j]
                               for j in range(num_materials))
        # min quality
        model.addConstr(lhs_expr >= quality_min[i] * demand[i],
                        name=f"quality_min_blend_{i}")
        # max quality
        model.addConstr(lhs_expr <= quality_max[i] * demand[i],
                        name=f"quality_max_blend_{i}")

    # Proportion (the old linear version)
    for i in range(num_blends):
        for j in range(num_materials):
            model.addConstr(
                x[i, j] <= p_max[j] * demand[i],
                name=f"pmax_blend_{i}_mat_{j}"
            )

    # -------------------------------------------
    # 6. Optimize
    # -------------------------------------------
    model.optimize()

    # -------------------------------------------
    # 7. Output
    # -------------------------------------------
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal objective (sum of squares) = {model.objVal:,.4f}")
        # Print only nonzero usage
        for i in range(num_blends):
            for j in range(num_materials):
                val = x[i, j].X
                if val > 1e-6:
                    print(f"Blend {i+1}, Material {j+1}: {val:.4f} tons")
    else:
        print(f"No optimal solution found. Status code: {model.status}")


if __name__ == "__main__":
    solve_ecoclean_blending_squares_objective()
