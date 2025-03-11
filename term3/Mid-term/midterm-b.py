import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_ecoclean_blending():
    # ------------------------------
    # 1. Read CSV data from GitHub
    # ------------------------------
    url_materials = ("https://raw.githubusercontent.com/"
                     "EthanRosehart/schulich_data_science/"
                     "refs/heads/main/term3/Mid-term/blending_materials.csv")
    url_blends = ("https://raw.githubusercontent.com/"
                  "EthanRosehart/schulich_data_science/"
                  "refs/heads/main/term3/Mid-term/blending_blends.csv")

    materials_df = pd.read_csv(url_materials)
    blends_df = pd.read_csv(url_blends)

    # We expect:
    # materials_df columns: [availability, cost, p_max]
    # blends_df columns:    [quality_1 ... quality_100, demand, quality_min, quality_max]

    num_materials = len(materials_df)  # should be 100
    num_blends = len(blends_df)        # should be 50

    # For convenience, store these in arrays for direct usage:
    availability = materials_df["availability"].to_list()  # length 100
    cost = materials_df["cost"].to_list()                  # length 100
    p_max = materials_df["p_max"].to_list()                # length 100

    demand = blends_df["demand"].to_list()                 # length 50
    quality_min = blends_df["quality_min"].to_list()       # length 50
    quality_max = blends_df["quality_max"].to_list()       # length 50

    # We'll collect each blend's "quality_j" contribution in a 2D list: quality[i][j].
    # Because the CSV has columns named quality_1, ..., quality_100:
    quality = []
    for i in range(num_blends):
        row_quality = []
        for j in range(1, num_materials+1):
            col_name = f"quality_{j}"
            val = blends_df.loc[i, col_name]  # quality_{j} for blend i
            row_quality.append(val)
        quality.append(row_quality)  # length=100 for each i

    # -------------------------------------------
    # 2. Create Gurobi Model and Decision Vars
    # -------------------------------------------
    model = gp.Model("EcoCleanBlending")

    # x[i,j] = tons of raw material j used in blend i
    # i in [0..num_blends-1], j in [0..num_materials-1]
    x = model.addVars(num_blends, num_materials,
                      lb=0, vtype=GRB.CONTINUOUS, name="x")

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

    # (a) Demand constraint: sum of raw materials for blend i = demand_i
    for i in range(num_blends):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(num_materials)) == demand[i],
            name=f"demand_of_blend_{i}"
        )

    # (b) Availability constraint: sum of usage of raw material j across blends <= availability_j
    for j in range(num_materials):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_blends)) <= availability[j],
            name=f"availability_of_mat_{j}"
        )

    # (c) Average Quality constraints:
    #     sum_j(quality[i][j] * x[i,j]) >= q_min[i] * demand[i]
    #     sum_j(quality[i][j] * x[i,j]) <= q_max[i] * demand[i]
    for i in range(num_blends):
        lhs_expr = gp.quicksum(quality[i][j] * x[i, j] for j in range(num_materials))
        model.addConstr(lhs_expr >= quality_min[i] * demand[i],
                        name=f"quality_min_blend_{i}")
        model.addConstr(lhs_expr <= quality_max[i] * demand[i],
                        name=f"quality_max_blend_{i}")

    # (d) Proportion constraint for each raw material:
    #     x[i,j] <= p_max[j] * demand[i]
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
        print(f"Optimal production cost: {model.objVal:,.2f}")
        # If desired, we can report how much of each material is used in each blend.
        # We'll only print non-zero amounts to avoid clutter.
        for i in range(num_blends):
            for j in range(num_materials):
                val = x[i, j].X
                if val > 1e-6:
                    print(f"Blend {i+1}, Material {j+1}: {val:.3f} tons")
                    
    else:
        print(f"No optimal solution found. Status code: {model.status}")

# Run the model-building and solving
if __name__ == "__main__":
    solve_ecoclean_blending()
