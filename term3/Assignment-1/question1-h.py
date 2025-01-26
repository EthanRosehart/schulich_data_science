"""
BioAgri Model: Region, Alpha%, and 50% Constraints (Parametric Analysis)
------------------------------------------------------------------------
We start with alpha=3.0% and reduce alpha in 0.1% steps.
Alpha replaces the 3% constraint:
   sum_{f} x[f,p] <= alpha * (total farm capacity)
We also keep:
   - Same-region constraint (y[p,c] = 0 if region mismatch)
   - 50% constraint (y[p,c] <= 0.5 * demand[c])

At each alpha, we solve the model and check feasibility.
We stop once we detect infeasibility (or reach our lower bound).
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def main():
    # --------------------------------------------------
    # 1. Load Data (same as your model)
    # --------------------------------------------------
    farms_df = pd.read_csv(
        'https://raw.githubusercontent.com/EthanRosehart/'
        'schulich_data_science/refs/heads/main/term3/Assignment-1/farms.csv'
    )
    plants_df = pd.read_csv(
        'https://raw.githubusercontent.com/EthanRosehart/'
        'schulich_data_science/refs/heads/main/term3/Assignment-1/processing.csv'
    )
    centers_df = pd.read_csv(
        'https://raw.githubusercontent.com/EthanRosehart/'
        'schulich_data_science/refs/heads/main/term3/Assignment-1/centers.csv'
    )

    # Create index lists
    farms = list(farms_df["Farm_ID"])                     
    plants = list(plants_df["Processing_Plant_ID"])        
    centers = list(centers_df["Center_ID"])                

    # Extract key data
    farm_capacity = {row.Farm_ID: row.Bio_Material_Capacity_Tons for row in farms_df.itertuples()}
    farm_purchase_cost = {row.Farm_ID: row.Cost_Per_Ton for row in farms_df.itertuples()}

    plant_capacity = {row.Processing_Plant_ID: row.Capacity_Tons for row in plants_df.itertuples()}
    plant_processing_cost = {row.Processing_Plant_ID: row.Processing_Cost_Per_Ton
                             for row in plants_df.itertuples()}
    plant_region = {row.Processing_Plant_ID: row.Region for row in plants_df.itertuples()}

    center_demand = {row.Center_ID: row.Requested_Demand_Tons for row in centers_df.itertuples()}
    center_region = {row.Center_ID: row.Region for row in centers_df.itertuples()}

    # Build transport cost dictionaries for farm->plant
    transport_farm_plant_cols = {f"Plant_{i}": f"Transport_Cost_To_Plant_{i}" for i in range(1, 19)}
    farm_to_plant_cost = {}
    for row in farms_df.itertuples():
        f_id = row.Farm_ID
        for p_id in plants:
            col_name = transport_farm_plant_cols[p_id]
            farm_to_plant_cost[(f_id, p_id)] = getattr(row, col_name)

    # Build transport cost dictionaries for plant->center
    transport_plant_center_cols = {f"Center_{j}": f"Transport_Cost_To_Center_{j}" for j in range(1, 103)}
    plant_to_center_cost = {}
    for row in plants_df.itertuples():
        p_id = row.Processing_Plant_ID
        for c_id in centers:
            col_name = transport_plant_center_cols[c_id]
            plant_to_center_cost[(p_id, c_id)] = getattr(row, col_name)

    # Total farm capacity for parametric constraint
    total_farm_capacity = sum(farm_capacity[f] for f in farms)

    # --------------------------------------------------
    # 2. Function to Build and Solve the Model with a given alpha
    # --------------------------------------------------
    def solve_model_with_alpha(alpha_decimal):
        """
        Builds and solves the model using 'alpha_decimal' instead of 3%.
        Returns (feasible, cost).
        """
        # Create Gurobi model
        model = gp.Model("BioAgri_ParametricAlpha")

        # Decision variables
        x = model.addVars(farms, plants, name="x", vtype=GRB.CONTINUOUS, lb=0)
        y = model.addVars(plants, centers, name="y", vtype=GRB.CONTINUOUS, lb=0)

        # Objective
        model.setObjective(
            gp.quicksum((farm_purchase_cost[f] + farm_to_plant_cost[(f,p)]) * x[f,p]
                        for f in farms for p in plants)
            + gp.quicksum((plant_processing_cost[p] + plant_to_center_cost[(p,c)]) * y[p,c]
                          for p in plants for c in centers),
            sense=GRB.MINIMIZE
        )

        # Base constraints
        # a) Farm capacity
        for f in farms:
            model.addConstr(
                gp.quicksum(x[f,p] for p in plants) <= farm_capacity[f],
                name=f"FarmCap_{f}"
            )

        # b) Plant capacity
        for p in plants:
            model.addConstr(
                gp.quicksum(x[f,p] for f in farms) <= plant_capacity[p],
                name=f"PlantCap_{p}"
            )

        # c) Flow conservation
        for p in plants:
            model.addConstr(
                gp.quicksum(y[p,c] for c in centers)
                <= gp.quicksum(x[f,p] for f in farms),
                name=f"FlowConserve_{p}"
            )

        # d) Center demand
        for c in centers:
            model.addConstr(
                gp.quicksum(y[p,c] for p in plants) == center_demand[c],
                name=f"Demand_{c}"
            )

        # e) Same-region: y[p,c] = 0 if mismatch
        for p in plants:
            for c in centers:
                if plant_region[p] != center_region[c]:
                    model.addConstr(y[p,c] == 0, name=f"SameRegion_{p}_{c}")

        # f) 50% constraint
        for c in centers:
            dem_c = center_demand[c]
            for p in plants:
                model.addConstr(
                    y[p,c] <= 0.5 * dem_c,
                    name=f"FiftyPercent_{p}_{c}"
                )

        # g) alpha constraint: sum_f x[f,p] <= alpha_decimal * total_farm_capacity
        for p in plants:
            model.addConstr(
                gp.quicksum(x[f,p] for f in farms) <= alpha_decimal * total_farm_capacity,
                name=f"Alpha_{p}"
            )

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return True, model.ObjVal
        else:
            return False, None

    # --------------------------------------------------
    # 3. Parametric Loop
    # --------------------------------------------------
    # Start at alpha=3.0%, go down to alpha=2.0% in 0.1% steps
    alpha_values = [i/10 for i in range(30, 19, -1)]  # [3.0, 2.9, 2.8, ... 2.0]
    best_feasible_alpha = None

    for alpha_percent in alpha_values:
        alpha_decimal = alpha_percent / 100.0
        feasible, cost = solve_model_with_alpha(alpha_decimal)
        if feasible:
            print(f"Alpha = {alpha_percent:.1f}%, Cost = {cost:,.2f} (Feasible)")
            best_feasible_alpha = alpha_percent
        else:
            print(f"Alpha = {alpha_percent:.1f}%, Infeasible.")
            # Once infeasible, typically we can stop searching further.
            break

    if best_feasible_alpha is not None:
        print(f"\nThe model becomes infeasible below about {best_feasible_alpha:.1f}%.\n")
    else:
        print("\nNo feasible solution found at or below 3.0%.\n")

if __name__ == "__main__":
    main()