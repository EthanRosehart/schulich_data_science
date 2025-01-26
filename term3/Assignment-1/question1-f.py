"""
BioAgri Model with Region, 3%, and 50% Constraints
--------------------------------------------------
We implement:
 - (1) Plant->Center only within the same region
 - (2) Each plant can process no more than 3% of total farm capacity
 - (3) Each plant cannot supply more than 50% of any single center's demand

We do not include the high-quality constraint.

Minimize:
  Sum( (FarmPurchase + Farm->PlantTransport)*x ) + Sum( (ProcessCost + Plant->CenterTransport)*y )

Constraints:
  1) Farm capacity
  2) Plant capacity
  3) Flow conservation (plant)
  4) Center demand
  5) Same-region constraint
  6) 3% constraint
  7) 50% constraint
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def main():
    # --------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------
    farms_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/farms.csv')
    plants_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/processing.csv')
    centers_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/centers.csv')

    # Create index lists
    farms = list(farms_df["Farm_ID"])                     # e.g. ["Farm_1", ..., "Farm_249"]
    plants = list(plants_df["Processing_Plant_ID"])        # e.g. ["Plant_1", ..., "Plant_18"]
    centers = list(centers_df["Center_ID"])                # e.g. ["Center_1", ..., "Center_102"]

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

    # --------------------------------------------------
    # 2. Initialize Gurobi Model
    # --------------------------------------------------
    model = gp.Model("BioAgri_CheapestConstraints")

    # --------------------------------------------------
    # 3. Decision Variables
    # --------------------------------------------------
    x = model.addVars(farms, plants, name="x", vtype=GRB.CONTINUOUS, lb=0)   # Farm->Plant flow
    y = model.addVars(plants, centers, name="y", vtype=GRB.CONTINUOUS, lb=0) # Plant->Center flow

    # --------------------------------------------------
    # 4. Objective Function
    # --------------------------------------------------
    model.setObjective(
        gp.quicksum((farm_purchase_cost[f] + farm_to_plant_cost[(f,p)]) * x[f,p]
                    for f in farms for p in plants)
        + gp.quicksum((plant_processing_cost[p] + plant_to_center_cost[(p,c)]) * y[p,c]
                      for p in plants for c in centers),
        sense=GRB.MINIMIZE
    )

    # --------------------------------------------------
    # 5. Base Constraints
    # --------------------------------------------------
    # (a) Farm capacity
    for f in farms:
        model.addConstr(
            gp.quicksum(x[f,p] for p in plants) <= farm_capacity[f],
            name=f"FarmCap_{f}"
        )

    # (b) Plant capacity
    for p in plants:
        model.addConstr(
            gp.quicksum(x[f,p] for f in farms) <= plant_capacity[p],
            name=f"PlantCap_{p}"
        )

    # (c) Flow conservation at plant
    for p in plants:
        model.addConstr(
            gp.quicksum(y[p,c] for c in centers)
            <= gp.quicksum(x[f,p] for f in farms),
            name=f"FlowConserve_{p}"
        )

    # (d) Center demand
    for c in centers:
        model.addConstr(
            gp.quicksum(y[p,c] for p in plants) == center_demand[c],
            name=f"Demand_{c}"
        )

    # --------------------------------------------------
    # 6. Additional Constraints
    # --------------------------------------------------
    # (1) Same-Region Constraint: If plant_region[p] != center_region[c], y[p,c] = 0
    for p in plants:
        for c in centers:
            if plant_region[p] != center_region[c]:
                model.addConstr(y[p,c] == 0, name=f"SameRegion_{p}_{c}")

    # (2) 3% Constraint: Each plant can process <= 3% of total farm capacity
    total_farm_capacity = sum(farm_capacity[f] for f in farms)
    for p in plants:
        model.addConstr(
            gp.quicksum(x[f,p] for f in farms) <= 0.03 * total_farm_capacity,
            name=f"ThreePercent_{p}"
        )

    # (3) 50% Constraint: A plant p cannot supply > 50% of a center c's demand
    for c in centers:
        dem_c = center_demand[c]
        for p in plants:
            model.addConstr(
                y[p,c] <= 0.5 * dem_c,
                name=f"FiftyPercent_{p}_{c}"
            )

    # --------------------------------------------------
    # 7. Solve
    # --------------------------------------------------
    model.optimize()

    # --------------------------------------------------
    # 8. Print Only the Cost
    # --------------------------------------------------
    if model.status == GRB.OPTIMAL:
        print(f"Optimal cost with region, 3%, and 50% constraints: {model.ObjVal:,.2f}")
    else:
        print(f"Model is infeasible or not optimal. Status: {model.status}")

if __name__ == "__main__":
    main()