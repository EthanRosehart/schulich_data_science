"""
BioAgri Risk Constraints Scenarios
----------------------------------
We define 5 scenarios:

1) Only 3% constraint (no high-quality, no same-region, no 50%).
2) Only 50% constraint (no high-quality, no same-region, no 3%).
3) Previous constraints (high-quality + same-region) + 3% constraint.
4) Previous constraints (high-quality + same-region) + 50% constraint.
5) All constraints (high-quality + same-region + 3% + 50%).

We only print the resulting optimal cost for each scenario.

"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# --------------------------------------------------
# Helper: Load Data Once
# --------------------------------------------------
def load_data():
    farms_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/farms.csv')
    plants_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/processing.csv')
    centers_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/centers.csv')

    # Index sets
    farms   = list(farms_df["Farm_ID"])
    plants  = list(plants_df["Processing_Plant_ID"])
    centers = list(centers_df["Center_ID"])

    # Farm data
    farm_capacity       = {row.Farm_ID: row.Bio_Material_Capacity_Tons for row in farms_df.itertuples()}
    farm_purchase_cost  = {row.Farm_ID: row.Cost_Per_Ton             for row in farms_df.itertuples()}
    farm_quality        = {row.Farm_ID: row.Quality                  for row in farms_df.itertuples()}

    # Plant data
    plant_capacity         = {row.Processing_Plant_ID: row.Capacity_Tons         for row in plants_df.itertuples()}
    plant_processing_cost  = {row.Processing_Plant_ID: row.Processing_Cost_Per_Ton for row in plants_df.itertuples()}
    plant_region           = {row.Processing_Plant_ID: row.Region                for row in plants_df.itertuples()}

    # Center data
    center_demand  = {row.Center_ID: row.Requested_Demand_Tons for row in centers_df.itertuples()}
    center_region  = {row.Center_ID: row.Region                for row in centers_df.itertuples()}

    # Build transport cost dictionaries for farm->plant
    # Example: "Transport_Cost_To_Plant_1", ..., "Transport_Cost_To_Plant_18"
    transport_farm_plant_cols = {f"Plant_{i}": f"Transport_Cost_To_Plant_{i}" for i in range(1,19)}
    farm_to_plant_cost = {}
    for row in farms_df.itertuples():
        f_id = row.Farm_ID
        for p_id in plants:
            col_name = transport_farm_plant_cols[p_id]
            farm_to_plant_cost[(f_id, p_id)] = getattr(row, col_name)

    # Build transport cost dictionaries for plant->center
    # Example: "Transport_Cost_To_Center_1", ..., "Transport_Cost_To_Center_102"
    transport_plant_center_cols = {f"Center_{j}": f"Transport_Cost_To_Center_{j}" for j in range(1,103)}
    plant_to_center_cost = {}
    for row in plants_df.itertuples():
        p_id = row.Processing_Plant_ID
        for c_id in centers:
            col_name = transport_plant_center_cols[c_id]
            plant_to_center_cost[(p_id, c_id)] = getattr(row, col_name)

    return {
        "farms": farms, "plants": plants, "centers": centers,
        "farm_capacity": farm_capacity,
        "farm_purchase_cost": farm_purchase_cost,
        "farm_quality": farm_quality,
        "plant_capacity": plant_capacity,
        "plant_processing_cost": plant_processing_cost,
        "plant_region": plant_region,
        "center_demand": center_demand,
        "center_region": center_region,
        "farm_to_plant_cost": farm_to_plant_cost,
        "plant_to_center_cost": plant_to_center_cost
    }

# --------------------------------------------------
# Helper: Build & Solve Model
# --------------------------------------------------
def build_and_solve(
    data,
    use_high_quality=False,
    use_same_region=False,
    use_3pct=False,
    use_50pct=False
):
    """
    Builds and solves the Gurobi model with specified constraints.
    Returns the objective value (or None if infeasible).
    """

    # Unpack data
    farms   = data["farms"]
    plants  = data["plants"]
    centers = data["centers"]

    farm_capacity        = data["farm_capacity"]
    farm_purchase_cost   = data["farm_purchase_cost"]
    farm_quality         = data["farm_quality"]

    plant_capacity       = data["plant_capacity"]
    plant_processing_cost= data["plant_processing_cost"]
    plant_region         = data["plant_region"]

    center_demand        = data["center_demand"]
    center_region        = data["center_region"]

    farm_to_plant_cost   = data["farm_to_plant_cost"]
    plant_to_center_cost = data["plant_to_center_cost"]

    # Create model
    model = gp.Model("BioAgri_RiskScenario")

    # Decision variables
    x = model.addVars(farms, plants, vtype=GRB.CONTINUOUS, name="x", lb=0)
    y = model.addVars(plants, centers, vtype=GRB.CONTINUOUS, name="y", lb=0)

    # Objective
    model.setObjective(
        gp.quicksum((farm_purchase_cost[f] + farm_to_plant_cost[(f,p)]) * x[f,p]
                    for f in farms for p in plants)
        + gp.quicksum((plant_processing_cost[p] + plant_to_center_cost[(p,c)]) * y[p,c]
                      for p in plants for c in centers),
        sense=GRB.MINIMIZE
    )

    # Base constraints
    # 1) Farm capacity
    for f in farms:
        model.addConstr(gp.quicksum(x[f,p] for p in plants) <= farm_capacity[f])

    # 2) Plant capacity
    for p in plants:
        model.addConstr(gp.quicksum(x[f,p] for f in farms) <= plant_capacity[p])

    # 3) Flow conservation at plant
    for p in plants:
        model.addConstr(gp.quicksum(y[p,c] for c in centers)
                        <= gp.quicksum(x[f,p] for f in farms))

    # 4) Center demand
    for c in centers:
        model.addConstr(gp.quicksum(y[p,c] for p in plants) == center_demand[c])

    # --------------------------------------------------
    # Additional Constraints
    # --------------------------------------------------
    # A) High-quality: if farm quality < 3, x[f,p] = 0
    if use_high_quality:
        for f in farms:
            if farm_quality[f] < 3:
                for p in plants:
                    model.addConstr(x[f,p] == 0)

    # B) Same-region: plant p can only send to center c if same region
    if use_same_region:
        for p in plants:
            for c in centers:
                if plant_region[p] != center_region[c]:
                    model.addConstr(y[p,c] == 0)

    # C) 3% limit: sum_f x[f,p] <= 0.03 * total_farm_capacity
    if use_3pct:
        total_farm_cap = sum(farm_capacity[f] for f in farms)
        for p in plants:
            model.addConstr(
                gp.quicksum(x[f,p] for f in farms) <= 0.03 * total_farm_cap,
                name=f"ThreePercent_{p}"
            )

    # D) 50% limit: y[p,c] <= 0.50 * center_demand[c]
    if use_50pct:
        for c in centers:
            demand_c = center_demand[c]
            for p in plants:
                model.addConstr(
                    y[p,c] <= 0.50 * demand_c,
                    name=f"FiftyPercent_{p}_{c}"
                )

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.ObjVal
    else:
        return None

# --------------------------------------------------
# 1) Scenario 1: Only the 3% constraint
# --------------------------------------------------
def scenario_1(data):
    obj_val = build_and_solve(
        data,
        use_high_quality=False,
        use_same_region=False,
        use_3pct=True,
        use_50pct=False
    )
    print(f"Scenario 1 (Only 3% constraint) Cost: {obj_val:,.2f}" if obj_val else "Scenario 1 Infeasible.")

# --------------------------------------------------
# 2) Scenario 2: Only the 50% constraint
# --------------------------------------------------
def scenario_2(data):
    obj_val = build_and_solve(
        data,
        use_high_quality=False,
        use_same_region=False,
        use_3pct=False,
        use_50pct=True
    )
    print(f"Scenario 2 (Only 50% constraint) Cost: {obj_val:,.2f}" if obj_val else "Scenario 2 Infeasible.")

# --------------------------------------------------
# 3) Scenario 3: Previous constraints (High-Quality + Same-Region) + 3% constraint
# --------------------------------------------------
def scenario_3(data):
    obj_val = build_and_solve(
        data,
        use_high_quality=True,   # previous constraints
        use_same_region=True,    # previous constraints
        use_3pct=True,           # new
        use_50pct=False
    )
    print(f"Scenario 3 (Prev + 3%) Cost: {obj_val:,.2f}" if obj_val else "Scenario 3 Infeasible.")

# --------------------------------------------------
# 4) Scenario 4: Previous constraints (High-Quality + Same-Region) + 50% constraint
# --------------------------------------------------
def scenario_4(data):
    obj_val = build_and_solve(
        data,
        use_high_quality=True,   # previous constraints
        use_same_region=True,    # previous constraints
        use_3pct=False,
        use_50pct=True           # new
    )
    print(f"Scenario 4 (Prev + 50%) Cost: {obj_val:,.2f}" if obj_val else "Scenario 4 Infeasible.")

# --------------------------------------------------
# 5) Scenario 5: All constraints (High-Quality + Same-Region + 3% + 50%)
# --------------------------------------------------
def scenario_5(data):
    obj_val = build_and_solve(
        data,
        use_high_quality=True,
        use_same_region=True,
        use_3pct=True,
        use_50pct=True
    )
    print(f"Scenario 5 (All constraints) Cost: {obj_val:,.2f}" if obj_val else "Scenario 5 Infeasible.")

# --------------------------------------------------
# Main: Run all 5 scenarios in sequence
# --------------------------------------------------
def main():
    data = load_data()
    scenario_1(data)
    scenario_2(data)
    scenario_3(data)
    scenario_4(data)
    scenario_5(data)

if __name__ == "__main__":
    main()