import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_ecoclean_blending_sum_of_squares():
    """
    Part (f): The amount of raw material j in blend i cannot exceed
              p_max[j] * (sum of squares of all x[i,k] in that blend).
              
    That is:   x[i,j] <= p_max[j] * sum_{k}( x[i,k]^2 ).

    This is no longer a purely linear program (it becomes a quadratic model).
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
    model = gp.Model("EcoCleanBlending_Nonlinear")

    # By default, Gurobi can handle some quadratic constraints.
    # If the model is recognized as non-convex, we might need:
    # model.Params.NonConvex = 2  # allow non-convex QCP if necessary

    # -------------------------------------------
    # 3. Decision Variables
    # -------------------------------------------
    # x[i,j] = tons of raw material j used in blend i
    x = model.addVars(num_blends, num_materials, lb=0,
                      vtype=GRB.CONTINUOUS, name="x")

    # sq[i,j] = (x[i,j])^2, introduced for sum-of-squares
    sq = model.addVars(num_blends, num_materials, lb=0,
                       vtype=GRB.CONTINUOUS, name="sq")

    # sqsum[i] = sum of squares of x[i,k] for k=1..100 in blend i
    sqsum = model.addVars(num_blends, lb=0, vtype=GRB.CONTINUOUS, name="sqsum")

    # -------------------------------------------
    # 4. Objective: Minimize total cost
    # -------------------------------------------
    model.setObjective(
        gp.quicksum(cost[j] * x[i, j]
                    for i in range(num_blends)
                    for j in range(num_materials)),
        GRB.MINIMIZE
    )

    # -------------------------------------------
    # 5. Constraints
    # -------------------------------------------

    # (a) Demand
    for i in range(num_blends):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(num_materials)) == demand[i],
            name=f"demand_of_blend_{i}"
        )

    # (b) Availability
    for j in range(num_materials):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_blends)) <= availability[j],
            name=f"availability_of_mat_{j}"
        )

    # (c) Quality constraints
    for i in range(num_blends):
        lhs_expr = gp.quicksum(quality[i][j] * x[i, j]
                               for j in range(num_materials))
        # min quality
        model.addConstr(lhs_expr >= quality_min[i] * demand[i],
                        name=f"quality_min_blend_{i}")
        # max quality
        model.addConstr(lhs_expr <= quality_max[i] * demand[i],
                        name=f"quality_max_blend_{i}")

    # (d) Define each sq[i,j] = x[i,j]^2 via Gurobi's built-in general power constraint
    for i in range(num_blends):
        for j in range(num_materials):
            # sq[i,j] = (x[i,j])^2
            model.addGenConstrPow(x[i,j], sq[i,j], 2.0, name=f"pow_{i}_{j}")

    # (e) Define sqsum[i] = sum_{j} sq[i,j]
    for i in range(num_blends):
        model.addConstr(
            sqsum[i] == gp.quicksum(sq[i,j] for j in range(num_materials)),
            name=f"sqsum_def_{i}"
        )

    # (f) Nonlinear proportion constraint:
    #     x[i,j] <= p_max[j] * sqsum[i]
    # (This replaces the old linear constraint x[i,j] <= p_max[j] * demand[i])
    for i in range(num_blends):
        for j in range(num_materials):
            model.addConstr(
                x[i,j] <= p_max[j] * sqsum[i],
                name=f"sum_of_squares_prop_{i}_{j}"
            )

    # -------------------------------------------
    # 6. Optimize
    # -------------------------------------------
    model.optimize()

    # -------------------------------------------
    # 7. Output
    # -------------------------------------------
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal nonlinear production cost: {model.objVal:,.2f}")
        # Print only nonzero usage
        for i in range(num_blends):
            for j in range(num_materials):
                val = x[i, j].X
                if val > 1e-6:
                    print(f"Blend {i+1}, Material {j+1}: {val:.4f} tons")
    else:
        print(f"No optimal solution found. Status code: {model.status}")

if __name__ == "__main__":
    solve_ecoclean_blending_sum_of_squares()
