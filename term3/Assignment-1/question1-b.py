# Assignment 1 - Question 1 - Part b)
# Ethan Rosehart - 221273420
# OMIS 6000 - Adam Diamant - Winter 2025

# Farms - Farm_ID,Bio_Material_Capacity_Tons,Quality,Cost_Per_Ton,Transport_Cost_To_Plant_1,Transport_Cost_To_Plant_2,Transport_Cost_To_Plant_3,Transport_Cost_To_Plant_4,Transport_Cost_To_Plant_5,Transport_Cost_To_Plant_6,Transport_Cost_To_Plant_7,Transport_Cost_To_Plant_8,Transport_Cost_To_Plant_9,Transport_Cost_To_Plant_10,Transport_Cost_To_Plant_11,Transport_Cost_To_Plant_12,Transport_Cost_To_Plant_13,Transport_Cost_To_Plant_14,Transport_Cost_To_Plant_15,Transport_Cost_To_Plant_16,Transport_Cost_To_Plant_17,Transport_Cost_To_Plant_18
# Processing - Processing_Plant_ID,Region,Capacity_Tons,Processing_Cost_Per_Ton,Transport_Cost_To_Center_1,Transport_Cost_To_Center_2,Transport_Cost_To_Center_3,Transport_Cost_To_Center_4,Transport_Cost_To_Center_5,Transport_Cost_To_Center_6,Transport_Cost_To_Center_7,Transport_Cost_To_Center_8,Transport_Cost_To_Center_9,Transport_Cost_To_Center_10,Transport_Cost_To_Center_11,Transport_Cost_To_Center_12,Transport_Cost_To_Center_13,Transport_Cost_To_Center_14,Transport_Cost_To_Center_15,Transport_Cost_To_Center_16,Transport_Cost_To_Center_17,Transport_Cost_To_Center_18,Transport_Cost_To_Center_19,Transport_Cost_To_Center_20,Transport_Cost_To_Center_21,Transport_Cost_To_Center_22,Transport_Cost_To_Center_23,Transport_Cost_To_Center_24,Transport_Cost_To_Center_25,Transport_Cost_To_Center_26,Transport_Cost_To_Center_27,Transport_Cost_To_Center_28,Transport_Cost_To_Center_29,Transport_Cost_To_Center_30,Transport_Cost_To_Center_31,Transport_Cost_To_Center_32,Transport_Cost_To_Center_33,Transport_Cost_To_Center_34,Transport_Cost_To_Center_35,Transport_Cost_To_Center_36,Transport_Cost_To_Center_37,Transport_Cost_To_Center_38,Transport_Cost_To_Center_39,Transport_Cost_To_Center_40,Transport_Cost_To_Center_41,Transport_Cost_To_Center_42,Transport_Cost_To_Center_43,Transport_Cost_To_Center_44,Transport_Cost_To_Center_45,Transport_Cost_To_Center_46,Transport_Cost_To_Center_47,Transport_Cost_To_Center_48,Transport_Cost_To_Center_49,Transport_Cost_To_Center_50,Transport_Cost_To_Center_51,Transport_Cost_To_Center_52,Transport_Cost_To_Center_53,Transport_Cost_To_Center_54,Transport_Cost_To_Center_55,Transport_Cost_To_Center_56,Transport_Cost_To_Center_57,Transport_Cost_To_Center_58,Transport_Cost_To_Center_59,Transport_Cost_To_Center_60,Transport_Cost_To_Center_61,Transport_Cost_To_Center_62,Transport_Cost_To_Center_63,Transport_Cost_To_Center_64,Transport_Cost_To_Center_65,Transport_Cost_To_Center_66,Transport_Cost_To_Center_67,Transport_Cost_To_Center_68,Transport_Cost_To_Center_69,Transport_Cost_To_Center_70,Transport_Cost_To_Center_71,Transport_Cost_To_Center_72,Transport_Cost_To_Center_73,Transport_Cost_To_Center_74,Transport_Cost_To_Center_75,Transport_Cost_To_Center_76,Transport_Cost_To_Center_77,Transport_Cost_To_Center_78,Transport_Cost_To_Center_79,Transport_Cost_To_Center_80,Transport_Cost_To_Center_81,Transport_Cost_To_Center_82,Transport_Cost_To_Center_83,Transport_Cost_To_Center_84,Transport_Cost_To_Center_85,Transport_Cost_To_Center_86,Transport_Cost_To_Center_87,Transport_Cost_To_Center_88,Transport_Cost_To_Center_89,Transport_Cost_To_Center_90,Transport_Cost_To_Center_91,Transport_Cost_To_Center_92,Transport_Cost_To_Center_93,Transport_Cost_To_Center_94,Transport_Cost_To_Center_95,Transport_Cost_To_Center_96,Transport_Cost_To_Center_97,Transport_Cost_To_Center_98,Transport_Cost_To_Center_99,Transport_Cost_To_Center_100,Transport_Cost_To_Center_101,Transport_Cost_To_Center_102
# Centers - Center_ID,Requested_Demand_Tons,Region

# Minimize total cost, satisfy the demand of each home center

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------

# Adjust file paths as needed
farms_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/farms.csv')
plants_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/processing.csv')
centers_df = pd.read_csv('https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/centers.csv')

# Create index lists
farms = list(farms_df['Farm_ID'])                     # e.g. ["Farm_1", ...]
plants = list(plants_df['Processing_Plant_ID'])        # e.g. ["Plant_1", "Plant_2", ...]
centers = list(centers_df['Center_ID'])                # e.g. ["Center_1", "Center_2", ...]

# Basic dictionaries
farm_capacity = {row.Farm_ID: row.Bio_Material_Capacity_Tons for row in farms_df.itertuples()}
farm_purchase_cost = {row.Farm_ID: row.Cost_Per_Ton for row in farms_df.itertuples()}

plant_capacity = {row.Processing_Plant_ID: row.Capacity_Tons for row in plants_df.itertuples()}
plant_processing_cost = {row.Processing_Plant_ID: row.Processing_Cost_Per_Ton for row in plants_df.itertuples()}

center_demand = {row.Center_ID: row.Requested_Demand_Tons for row in centers_df.itertuples()}

# --------------------------------------------------
# 2. Build column mappings to read transport costs
# --------------------------------------------------

# "Plant_1" -> "Transport_Cost_To_Plant_1"
transport_farm_plant_cols = {f"Plant_{i}": f"Transport_Cost_To_Plant_{i}" for i in range(1, 19)}

# "Center_1" -> "Transport_Cost_To_Center_1"
transport_plant_center_cols = {f"Center_{j}": f"Transport_Cost_To_Center_{j}" for j in range(1, 103)}

farm_to_plant_cost = {}
for row in farms_df.itertuples():
    f_id = row.Farm_ID
    for p_id in plants:
        col_name = transport_farm_plant_cols[p_id]
        farm_to_plant_cost[(f_id, p_id)] = getattr(row, col_name)

plant_to_center_cost = {}
for row in plants_df.itertuples():
    p_id = row.Processing_Plant_ID
    for c_id in centers:
        col_name = transport_plant_center_cols[c_id]
        plant_to_center_cost[(p_id, c_id)] = getattr(row, col_name)

# --------------------------------------------------
# 3. Build and Solve Model
# --------------------------------------------------
model = gp.Model('BioAgri_MinCost')

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

# Constraints
# Farm capacity
for f in farms:
    model.addConstr(gp.quicksum(x[f,p] for p in plants) <= farm_capacity[f], name=f"FarmCap_{f}")

# Plant capacity
for p in plants:
    model.addConstr(gp.quicksum(x[f,p] for f in farms) <= plant_capacity[p], name=f"PlantCap_{p}")

# Flow conservation at plant
for p in plants:
    model.addConstr(gp.quicksum(y[p,c] for c in centers) <= gp.quicksum(x[f,p] for f in farms),
                    name=f"FlowConserve_{p}")

# Demand at each center
for c in centers:
    model.addConstr(gp.quicksum(y[p,c] for p in plants) == center_demand[c], name=f"Demand_{c}")

model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value (Min Cost) = {model.ObjVal:,.2f}")
else:
    print(f"Model did not solve to optimality. Status: {model.status}")