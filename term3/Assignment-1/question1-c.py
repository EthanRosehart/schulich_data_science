# Assignment 1 - Part c
# Ethan Rosehart - 221273420
# Additional Constraint: Processing plants are restricted to only send fertilizer to home centers in the same region

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
    # Example: ["Farm_1", "Farm_2", ..., "Farm_249"]
    farms = list(farms_df['Farm_ID'])
    # Example: ["Plant_1", "Plant_2", ..., "Plant_18"]
    plants = list(plants_df['Processing_Plant_ID'])
    # Example: ["Center_1", "Center_2", ..., "Center_102"]
    centers = list(centers_df['Center_ID'])

    # --------------------------------------------------
    # 2. Extract Key Data
    # --------------------------------------------------
    # Farm data
    farm_capacity = {row.Farm_ID: row.Bio_Material_Capacity_Tons 
                     for row in farms_df.itertuples()}
    farm_purchase_cost = {row.Farm_ID: row.Cost_Per_Ton 
                          for row in farms_df.itertuples()}

    # Processing plant data
    plant_capacity = {row.Processing_Plant_ID: row.Capacity_Tons 
                      for row in plants_df.itertuples()}
    plant_processing_cost = {row.Processing_Plant_ID: row.Processing_Cost_Per_Ton 
                             for row in plants_df.itertuples()}
    plant_region = {row.Processing_Plant_ID: row.Region 
                    for row in plants_df.itertuples()}

    # Centers data
    center_demand = {row.Center_ID: row.Requested_Demand_Tons 
                     for row in centers_df.itertuples()}
    center_region = {row.Center_ID: row.Region 
                     for row in centers_df.itertuples()}

    # --------------------------------------------------
    # 3. Create Transport Cost Dictionaries
    # --------------------------------------------------
    # Because the columns are named "Transport_Cost_To_Plant_1", ..., "Transport_Cost_To_Plant_18"
    # and the plant IDs are "Plant_1", "Plant_2", ..., we make a mapping:
    transport_farm_plant_cols = {
        f"Plant_{i}": f"Transport_Cost_To_Plant_{i}" for i in range(1, 19)
    }

    # Because the columns for center transport are named "Transport_Cost_To_Center_1", ..., "Transport_Cost_To_Center_102"
    # and the center IDs are "Center_1", "Center_2", ..., we make a mapping:
    transport_plant_center_cols = {
        f"Center_{j}": f"Transport_Cost_To_Center_{j}" for j in range(1, 103)
    }

    # Build the dictionary for Farm->Plant costs
    farm_to_plant_cost = {}
    for row in farms_df.itertuples():
        f_id = row.Farm_ID
        for p_id in plants:
            # Map "Plant_5" -> "Transport_Cost_To_Plant_5"
            col_name = transport_farm_plant_cols[p_id]
            farm_to_plant_cost[(f_id, p_id)] = getattr(row, col_name)

    # Build the dictionary for Plant->Center costs
    plant_to_center_cost = {}
    for row in plants_df.itertuples():
        p_id = row.Processing_Plant_ID
        for c_id in centers:
            col_name = transport_plant_center_cols[c_id]
            plant_to_center_cost[(p_id, c_id)] = getattr(row, col_name)

    # --------------------------------------------------
    # 4. Initialize Gurobi Model
    # --------------------------------------------------
    model = gp.Model("BioAgri_MinCost_Regional")

    # --------------------------------------------------
    # 5. Decision Variables
    # --------------------------------------------------
    # x[f, p] = flow of raw material from farm f to plant p
    x = model.addVars(farms, plants, name="x", vtype=GRB.CONTINUOUS, lb=0)

    # y[p, c] = flow of processed fertilizer from plant p to center c
    y = model.addVars(plants, centers, name="y", vtype=GRB.CONTINUOUS, lb=0)

    # --------------------------------------------------
    # 6. Objective Function
    # --------------------------------------------------
    #   SUM (Farm cost + Farm->Plant) * x  +  SUM (Processing cost + Plant->Center) * y
    model.setObjective(
        gp.quicksum(
            (farm_purchase_cost[f] + farm_to_plant_cost[(f, p)]) * x[f, p]
            for f in farms for p in plants
        )
        + gp.quicksum(
            (plant_processing_cost[p] + plant_to_center_cost[(p, c)]) * y[p, c]
            for p in plants for c in centers
        ),
        sense=GRB.MINIMIZE
    )

    # --------------------------------------------------
    # 7. Constraints
    # --------------------------------------------------
    # a) Farm capacity: sum of raw shipped <= farm capacity
    for f in farms:
        model.addConstr(gp.quicksum(x[f, p] for p in plants) <= farm_capacity[f],
                        name=f"FarmCap_{f}")

    # b) Plant capacity: sum of raw received <= plant capacity
    for p in plants:
        model.addConstr(gp.quicksum(x[f, p] for f in farms) <= plant_capacity[p],
                        name=f"PlantCap_{p}")

    # c) Flow conservation at plant: output <= input
    for p in plants:
        model.addConstr(gp.quicksum(y[p, c] for c in centers)
                        <= gp.quicksum(x[f, p] for f in farms),
                        name=f"FlowConserve_{p}")

    # d) Center demand: sum of inputs = demand
    for c in centers:
        model.addConstr(gp.quicksum(y[p, c] for p in plants) == center_demand[c],
                        name=f"Demand_{c}")

    # e) **Regional Restriction**: plant p can only send to center c if they share the same region
    for p in plants:
        for c in centers:
            if plant_region[p] != center_region[c]:
                model.addConstr(y[p, c] == 0,
                                name=f"RegionConstraint_{p}_{c}")

    # --------------------------------------------------
    # 8. Solve the Model
    # --------------------------------------------------
    model.optimize()

    # --------------------------------------------------
    # 9. Print Results
    # --------------------------------------------------
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value (Min Cost) = {model.ObjVal:,.2f}")
        print("\nNon-zero flows (Farm -> Plant):")
        for f in farms:
            for p in plants:
                flow_x = x[f, p].X
                if flow_x > 1e-6:
                    print(f"  {f} -> {p}: {flow_x:.1f} tons")
        print("\nNon-zero flows (Plant -> Center):")
        for p in plants:
            for c in centers:
                flow_y = y[p, c].X
                if flow_y > 1e-6:
                    print(f"  {p} -> {c}: {flow_y:.1f} tons")
    else:
        print(f"Model did not solve to optimality. Status: {model.status}")


if __name__ == "__main__":
    main()
