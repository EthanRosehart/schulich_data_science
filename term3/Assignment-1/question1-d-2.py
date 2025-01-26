# Question 1 - Part D - Does not include region constraint

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

    # Create index lists (string IDs)
    farms = list(farms_df['Farm_ID'])                     # e.g., ["Farm_1", ..., "Farm_249"]
    plants = list(plants_df['Processing_Plant_ID'])        # e.g., ["Plant_1", ..., "Plant_18"]
    centers = list(centers_df['Center_ID'])                # e.g., ["Center_1", ..., "Center_102"]

    # --------------------------------------------------
    # 2. Extract Key Data
    # --------------------------------------------------
    # Farm data
    farm_capacity = {row.Farm_ID: row.Bio_Material_Capacity_Tons for row in farms_df.itertuples()}
    farm_purchase_cost = {row.Farm_ID: row.Cost_Per_Ton for row in farms_df.itertuples()}
    farm_quality = {row.Farm_ID: row.Quality for row in farms_df.itertuples()}

    # Processing plant data
    plant_capacity = {row.Processing_Plant_ID: row.Capacity_Tons for row in plants_df.itertuples()}
    plant_processing_cost = {row.Processing_Plant_ID: row.Processing_Cost_Per_Ton
                             for row in plants_df.itertuples()}

    # Centers data
    center_demand = {row.Center_ID: row.Requested_Demand_Tons for row in centers_df.itertuples()}

    # --------------------------------------------------
    # 3. Create Transport Cost Dictionaries
    # --------------------------------------------------
    # Because columns are named e.g. "Transport_Cost_To_Plant_1", ..., "Transport_Cost_To_Plant_18"
    # and the processing plants are named "Plant_1", ..., "Plant_18", we map them:
    transport_farm_plant_cols = {
        f"Plant_{i}": f"Transport_Cost_To_Plant_{i}" for i in range(1, 19)
    }

    # Because columns are named "Transport_Cost_To_Center_1", ..., "Transport_Cost_To_Center_102"
    # and centers are "Center_1", ..., "Center_102", we map them similarly:
    transport_plant_center_cols = {
        f"Center_{j}": f"Transport_Cost_To_Center_{j}" for j in range(1, 103)
    }

    # Build the dictionary for Farm->Plant costs
    farm_to_plant_cost = {}
    for row in farms_df.itertuples():
        f_id = row.Farm_ID
        for p_id in plants:
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
    model = gp.Model("BioAgri_MinCost_HighQuality")

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
    # sum_{f,p} [purchase_cost(f) + farm_to_plant_cost(f,p)] * x[f,p]
    # + sum_{p,c} [processing_cost(p) + plant_to_center_cost(p,c)] * y[p,c]
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
    # (a) Farm capacity
    for f in farms:
        model.addConstr(
            gp.quicksum(x[f, p] for p in plants) <= farm_capacity[f],
            name=f"FarmCap_{f}"
        )

    # (b) Plant capacity
    for p in plants:
        model.addConstr(
            gp.quicksum(x[f, p] for f in farms) <= plant_capacity[p],
            name=f"PlantCap_{p}"
        )

    # (c) Flow conservation at plants
    for p in plants:
        model.addConstr(
            gp.quicksum(y[p, c] for c in centers)
            <= gp.quicksum(x[f, p] for f in farms),
            name=f"FlowConserve_{p}"
        )

    # (d) Center demand
    for c in centers:
        model.addConstr(
            gp.quicksum(y[p, c] for p in plants) == center_demand[c],
            name=f"Demand_{c}"
        )

    # (e) **Quality Restriction**: If farm's quality < 3, x[f,p] = 0
    for f in farms:
        if farm_quality[f] < 3:
            for p in plants:
                model.addConstr(
                    x[f, p] == 0,
                    name=f"HighQualityOnly_{f}_{p}"
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
        print("\nNon-zero flows (Farm -> Plant):")
        for f in farms:
            for p in plants:
                flow_x = x[f, p].X
                if flow_x > 1e-6:  # threshold to ignore near-zero
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