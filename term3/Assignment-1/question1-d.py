"""
BioAgri Model with Both Constraints:
-----------------------------------
1) Only farms with Quality >= 3 can supply raw material (Quality < 3 => x[f,p] = 0).
2) A processing plant can only send fertilizer to centers in the same region 
   (plant_region[p] != center_region[c] => y[p,c] = 0).

We minimize the total cost = 
  Raw material cost + transport (farm->plant) + processing cost + transport (plant->center).

Constraints:
  - Farm capacity
  - Plant capacity
  - Plant flow conservation
  - Center demand
  - Quality restriction (farms)
  - Region restriction (plants->centers)
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
    farms = list(farms_df['Farm_ID'])                    # e.g. ["Farm_1", ..., "Farm_249"]
    plants = list(plants_df['Processing_Plant_ID'])       # e.g. ["Plant_1", ..., "Plant_18"]
    centers = list(centers_df['Center_ID'])               # e.g. ["Center_1", ..., "Center_102"]

    # --------------------------------------------------
    # 2. Extract Key Data
    # --------------------------------------------------
    # Farm info
    farm_capacity = {row.Farm_ID: row.Bio_Material_Capacity_Tons for row in farms_df.itertuples()}
    farm_purchase_cost = {row.Farm_ID: row.Cost_Per_Ton for row in farms_df.itertuples()}
    farm_quality = {row.Farm_ID: row.Quality for row in farms_df.itertuples()}

    # Plant info
    plant_capacity = {row.Processing_Plant_ID: row.Capacity_Tons for row in plants_df.itertuples()}
    plant_processing_cost = {row.Processing_Plant_ID: row.Processing_Cost_Per_Ton 
                             for row in plants_df.itertuples()}
    plant_region = {row.Processing_Plant_ID: row.Region for row in plants_df.itertuples()}

    # Center info
    center_demand = {row.Center_ID: row.Requested_Demand_Tons for row in centers_df.itertuples()}
    center_region = {row.Center_ID: row.Region for row in centers_df.itertuples()}

    # --------------------------------------------------
    # 3. Create Transport Cost Dictionaries
    # --------------------------------------------------
    # Build mapping from "Plant_1" -> "Transport_Cost_To_Plant_1", etc.
    transport_farm_plant_cols = {
        f"Plant_{i}": f"Transport_Cost_To_Plant_{i}" for i in range(1, 19)
    }

    # Build mapping from "Center_1" -> "Transport_Cost_To_Center_1", etc.
    transport_plant_center_cols = {
        f"Center_{j}": f"Transport_Cost_To_Center_{j}" for j in range(1, 103)
    }

    # Dictionary for Farm->Plant costs
    farm_to_plant_cost = {}
    for row in farms_df.itertuples():
        f_id = row.Farm_ID
        for p_id in plants:
            col_name = transport_farm_plant_cols[p_id]
            farm_to_plant_cost[(f_id, p_id)] = getattr(row, col_name)

    # Dictionary for Plant->Center costs
    plant_to_center_cost = {}
    for row in plants_df.itertuples():
        p_id = row.Processing_Plant_ID
        for c_id in centers:
            col_name = transport_plant_center_cols[c_id]
            plant_to_center_cost[(p_id, c_id)] = getattr(row, col_name)

    # --------------------------------------------------
    # 4. Initialize Gurobi Model
    # --------------------------------------------------
    model = gp.Model("BioAgri_HighQuality_Regional")

    # --------------------------------------------------
    # 5. Decision Variables
    # --------------------------------------------------
    # x[f,p] = flow of raw material from farm f to plant p
    x = model.addVars(farms, plants, name="x", vtype=GRB.CONTINUOUS, lb=0)

    # y[p,c] = flow of processed fertilizer from plant p to center c
    y = model.addVars(plants, centers, name="y", vtype=GRB.CONTINUOUS, lb=0)

    # --------------------------------------------------
    # 6. Objective Function
    # --------------------------------------------------
    model.setObjective(
        # Farm purchase + farm->plant transport
        gp.quicksum(
            (farm_purchase_cost[f] + farm_to_plant_cost[(f, p)]) * x[f, p]
            for f in farms for p in plants
        )
        # Plant processing + plant->center transport
        + gp.quicksum(
            (plant_processing_cost[p] + plant_to_center_cost[(p, c)]) * y[p, c]
            for p in plants for c in centers
        ),
        sense=GRB.MINIMIZE
    )

    # --------------------------------------------------
    # 7. Constraints
    # --------------------------------------------------
    # a) Farm capacity
    for f in farms:
        model.addConstr(
            gp.quicksum(x[f, p] for p in plants) <= farm_capacity[f],
            name=f"FarmCap_{f}"
        )

    # b) Plant capacity
    for p in plants:
        model.addConstr(
            gp.quicksum(x[f, p] for f in farms) <= plant_capacity[p],
            name=f"PlantCap_{p}"
        )

    # c) Flow conservation at plants
    for p in plants:
        model.addConstr(
            gp.quicksum(y[p, c] for c in centers)
            <= gp.quicksum(x[f, p] for f in farms),
            name=f"FlowConserve_{p}"
        )

    # d) Center demand
    for c in centers:
        model.addConstr(
            gp.quicksum(y[p, c] for p in plants) == center_demand[c],
            name=f"Demand_{c}"
        )

    # e) **Quality Restriction**: If a farm's quality < 3, x[f,p] = 0
    for f in farms:
        if farm_quality[f] < 3:
            for p in plants:
                model.addConstr(
                    x[f, p] == 0,
                    name=f"HighQualityOnly_{f}_{p}"
                )

    # f) **Regional Restriction**: plant -> center only if same region
    for p in plants:
        for c in centers:
            if plant_region[p] != center_region[c]:
                model.addConstr(
                    y[p, c] == 0,
                    name=f"RegionConstraint_{p}_{c}"
                )

    # --------------------------------------------------
    # 8. Solve Model
    # --------------------------------------------------
    model.optimize()

    # --------------------------------------------------
    # 9. Print Results
    # --------------------------------------------------
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value (Min Cost) = {model.ObjVal:,.2f}")
        
        print("\nNon-zero flows (Farm -> Plant) [Quality >= 3 only]:")
        for f in farms:
            for p in plants:
                flow_x = x[f, p].X
                if flow_x > 1e-6:  # filter out zero flows
                    print(f"  {f} -> {p}: {flow_x:.2f} tons")

        print("\nNon-zero flows (Plant -> Center) [Same region only]:")
        for p in plants:
            for c in centers:
                flow_y = y[p, c].X
                if flow_y > 1e-6:
                    print(f"  {p} -> {c}: {flow_y:.2f} tons")
    else:
        print(f"Model did not solve to optimality. Status: {model.status}")


if __name__ == "__main__":
    main()