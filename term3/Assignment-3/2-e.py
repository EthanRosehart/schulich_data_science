# Question 2 - Part e) Optimal Capacity and Cost with 20x10 using SAA

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import random

def read_costs_csv(url):
    """
    Reads costs.csv with an extra comma.
    Renames 'Unnamed: 0' => 'Station', sets as index.
    Ensures row/column labels are strings like 'Station_0'..'Station_14'.
    Converts each cell to float if possible.
    """
    df = pd.read_csv(url, header=0)
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "Station"}, inplace=True)
    df.set_index("Station", inplace=True)

    # Convert each cell to float
    df = df.apply(pd.to_numeric, errors="coerce")

    # Strip whitespace
    df.index   = df.index.str.strip()
    df.columns = df.columns.str.strip()

    # Debug prints
    print("\n--- Debug: df_costs shape, index, columns ---")
    print("shape:", df.shape)
    print("index:", df.index)
    print("cols: ", df.columns)
    print("---------------------------------------------\n")

    return df

def read_randomness_csv(url):
    """
    Reads randomness.csv similarly.
    """
    df = pd.read_csv(url, header=0)
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "Station"}, inplace=True)
    df.set_index("Station", inplace=True)

    df = df.apply(pd.to_numeric, errors="coerce")

    df.index = df.index.str.strip()
    df.columns = df.columns.str.strip()

    # Debug
    print("\n--- Debug: df_rand shape, index, columns ---")
    print("shape:", df.shape)
    print("index:", df.index)
    print("cols: ", df.columns)
    print("---------------------------------------------\n")

    return df

def main():
    # Basic SAA setup
    NUM_TRIALS = 20
    SCENARIOS_PER_TRIAL = 10
    SCENARIO_COUNT = NUM_TRIALS * SCENARIOS_PER_TRIAL
    random_seed = 12345
    random.seed(random_seed)
    np.random.seed(random_seed)

    cost_over  = 0.09
    cost_under = 0.13

    # 1) Read CSVs
    costs_url = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-3/costs.csv"
    df_costs  = read_costs_csv(costs_url)

    rand_url  = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-3/randomness.csv"
    df_rand   = read_randomness_csv(rand_url)

    # Build dictionaries from df_rand
    p_need   = {}
    mean_dem = {}
    std_dem  = {}

    for st in df_rand.index:  # 'Station_1'..'Station_14'
        p_need[st]   = df_rand.loc[st,"Probability"]
        mean_dem[st] = df_rand.loc[st,"Mean_Demand"]
        std_dem[st]  = df_rand.loc[st,"Std_Dev_Demand"]

    # We'll interpret Station_0 as depot
    all_stations = [f"Station_{i}" for i in range(1,15)]  # stations that have demand
    all_nodes    = [f"Station_{i}" for i in range(15)]   # 0..14

    # 2) Build route_cost, forcibly cast to float, with debugging
    route_cost = {}
    for i in all_nodes:
        if i not in df_costs.index:
            print(f"WARNING: Row label '{i}' not found in df_costs.index = {list(df_costs.index)}")
        for j in all_nodes:
            if j not in df_costs.columns:
                print(f"WARNING: Col label '{j}' not found in df_costs.columns = {list(df_costs.columns)}")

            val = df_costs.loc[i, j] if (i in df_costs.index and j in df_costs.columns) else None
            try:
                route_cost[(i,j)] = float(val)  # forcibly cast
            except:
                route_cost[(i,j)] = 9999999.0
                print(f"WARNING: Could not cast df_costs.loc[{i},{j}]='{val}' to float. Setting it to 9999999.0")

    # 3) Generate SCENARIO_COUNT random scenarios
    scenario_list = list(range(SCENARIO_COUNT))
    scenario_prob = {s: 1.0/SCENARIO_COUNT for s in scenario_list}

    stations_needed_dict = {}
    demand_dict = {}

    for s in scenario_list:
        needed_stations = []
        tot_dem = 0.0
        for st in all_stations:
            if random.random() < p_need[st]:
                d = np.random.normal(mean_dem[st], std_dem[st])
                if d < 0:
                    d = 0.0
                needed_stations.append(st)
                tot_dem += d
        stations_needed_dict[s] = needed_stations
        demand_dict[s] = tot_dem

    # 4) Build single MILP
    m = gp.Model("StochasticSingleVehicleVRP")
    K = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="TruckCapacity")

    x = {}
    surplus = {}
    shortfall = {}

    for s in scenario_list:
        nodes_s = ["Station_0"] + stations_needed_dict[s]
        if len(nodes_s)>1:
            for i in nodes_s:
                for j in nodes_s:
                    if i!=j:
                        x[s,i,j] = m.addVar(vtype=GRB.BINARY, name=f"x_s{s}_{i}_{j}")
        surplus[s]   = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"surplus_s{s}")
        shortfall[s] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"shortfall_s{s}")

    # Objective
    obj_expr = gp.LinExpr()
    for s in scenario_list:
        prob_s = scenario_prob[s]
        route_expr = gp.LinExpr()
        nodes_s = ["Station_0"] + stations_needed_dict[s]
        if len(nodes_s)>1:
            for i in nodes_s:
                for j in nodes_s:
                    if i!=j:
                        cost_ij = route_cost[(i,j)]  # float
                        route_expr += cost_ij * x[s,i,j]
        mismatch_expr = cost_over*surplus[s] + cost_under*shortfall[s]
        obj_expr.add(prob_s * (route_expr + mismatch_expr))

    m.setObjective(obj_expr, GRB.MINIMIZE)

    # Constraints
    for s in scenario_list:
        nodes_s = ["Station_0"] + stations_needed_dict[s]
        if len(nodes_s)>1:
            needed = stations_needed_dict[s]
            # in-degree/out-degree=1 for each needed station
            for st in needed:
                m.addConstr(
                    gp.quicksum(x[s,i,st] for i in nodes_s if i!=st) == 1,
                    name=f"In_s{s}_{st}"
                )
                m.addConstr(
                    gp.quicksum(x[s,st,j] for j in nodes_s if j!=st) == 1,
                    name=f"Out_s{s}_{st}"
                )
            # depot in/out=1 if needed>0
            if len(needed)>0:
                m.addConstr(
                    gp.quicksum(x[s,"Station_0",j] for j in nodes_s if j!="Station_0")==1,
                    name=f"DepotOut_s{s}"
                )
                m.addConstr(
                    gp.quicksum(x[s,i,"Station_0"] for i in nodes_s if i!="Station_0")==1,
                    name=f"DepotIn_s{s}"
                )
            # Subtour elimination
            if len(needed)>1:
                u_s = m.addVars(needed, lb=0, ub=len(needed), name=f"u_s{s}")
                for i in needed:
                    for j in needed:
                        if i!=j:
                            m.addConstr(
                                u_s[i]-u_s[j]+(len(needed))*x[s,i,j] <= len(needed)-1,
                                name=f"MTZ_s{s}_{i}_{j}"
                            )
        # Surplus/Shortfall
        D_s = demand_dict[s]
        m.addConstr(surplus[s]>=K - D_s, name=f"Surplus_s{s}")
        m.addConstr(shortfall[s]>=D_s - K, name=f"Shortfall_s{s}")

    # Solve
    m.setParam("OutputFlag", 1)
    m.optimize()

    # Results
    if m.status==GRB.OPTIMAL:
        print("\n=== Optimal solution found ===")
        print(f"Truck capacity K = {K.X:,.2f}")
        print(f"Optimal expected cost = {m.ObjVal:,.2f}")
    else:
        print("No optimal solution or model not optimal.")

if __name__=="__main__":
    main()
