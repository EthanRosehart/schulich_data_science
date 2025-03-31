# Question 2: Part e and f - Use part e to find EVPI and VSS with EV solution

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import random

###############################################################################
# 1) Read CSV files
###############################################################################
def read_costs_csv(url):
    df = pd.read_csv(url, header=0)
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0":"Station"}, inplace=True)
    df.set_index("Station", inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")  # ensure numeric
    df.index   = df.index.str.strip()
    df.columns = df.columns.str.strip()
    return df

def read_randomness_csv(url):
    df = pd.read_csv(url, header=0)
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0":"Station"}, inplace=True)
    df.set_index("Station", inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index   = df.index.str.strip()
    df.columns = df.columns.str.strip()
    return df

###############################################################################
# 2) Solve a daily MIP (single scenario) for Wait & See
###############################################################################
def solve_daily_mip(df_costs, demands, cost_over, cost_under):
    """
    Wait & See approach for a single scenario:
      - We pick truck capacity K, route among whichever stations have demands>0
      - Minimizes route cost + mismatch cost (over/under).
    demands: dictionary station-> demand for this day (0 for stations that don't need fueling)
    returns minimal cost for that scenario
    """
    m = gp.Model("DailyScenarioMIP")
    m.setParam('OutputFlag', 0)

    # define the set of stations that actually need fueling
    needed_stations = [st for st in demands if demands[st] > 0]
    nodes = ["Station_0"] + needed_stations

    # capacity K (continuous)
    K = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="K_scenario")

    # Surplus/Shortfall
    total_dem = sum(demands.values())
    surplus   = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="surplus")
    shortfall = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="shortfall")

    # route variables if we have >1 node
    x = {}
    if len(nodes) > 1:
        for i in nodes:
            for j in nodes:
                if i!=j:
                    x[i,j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # objective = route cost + mismatch cost
    route_expr = gp.LinExpr()
    if len(nodes)>1:
        for i in nodes:
            for j in nodes:
                if i!=j:
                    cost_ij = df_costs.loc[i,j]
                    route_expr += cost_ij * x[i,j]

    mismatch_expr = cost_over*surplus + cost_under*shortfall
    m.setObjective(route_expr + mismatch_expr, GRB.MINIMIZE)

    # constraints
    # capacity => surplus >= K - total_dem
    # shortfall >= total_dem - K
    m.addConstr(surplus   >= K - total_dem)
    m.addConstr(shortfall >= total_dem - K)

    if len(nodes)>1:
        # in-degree, out-degree=1 for each needed station
        for st in needed_stations:
            m.addConstr(gp.quicksum(x[i,st] for i in nodes if i!=st)==1)
            m.addConstr(gp.quicksum(x[st,j] for j in nodes if j!=st)==1)
        # depot in/out=1 if we have needed stations
        if len(needed_stations)>0:
            m.addConstr(gp.quicksum(x["Station_0",j] 
                                    for j in nodes if j!="Station_0")==1)
            m.addConstr(gp.quicksum(x[i,"Station_0"] 
                                    for i in nodes if i!="Station_0")==1)
        # subtour elimination if >1 station
        if len(needed_stations)>1:
            u = m.addVars(needed_stations, lb=0, ub=len(needed_stations))
            for i in needed_stations:
                for j in needed_stations:
                    if i!=j:
                        m.addConstr(u[i]-u[j]+(len(needed_stations))*x[i,j]
                                    <= len(needed_stations)-1)

    # solve
    m.optimize()
    if m.status==GRB.OPTIMAL:
        return m.ObjVal
    else:
        return 1e9

###############################################################################
# 3) Solve a TSP that visits all stations for the Mean Value solution
###############################################################################
def solve_tsp_all_stations(df_costs):
    m = gp.Model("AllStationsTSP")
    m.setParam('OutputFlag',0)
    # stations = Station_1..14 plus depot Station_0
    stations_14 = [f"Station_{i}" for i in range(1,15)]
    nodes = ["Station_0"] + stations_14

    x = m.addVars(nodes, nodes, vtype=GRB.BINARY, name="x")
    for i in nodes:
        x[i,i].ub=0

    obj_expr = gp.LinExpr()
    for i in nodes:
        for j in nodes:
            if i!=j:
                obj_expr += df_costs.loc[i,j]*x[i,j]

    m.setObjective(obj_expr, GRB.MINIMIZE)

    # in-degree/out-degree=1 for each station
    for st in stations_14:
        m.addConstr(gp.quicksum(x[i,st] for i in nodes if i!=st)==1)
        m.addConstr(gp.quicksum(x[st,j] for j in nodes if j!=st)==1)
    m.addConstr(gp.quicksum(x["Station_0", j] 
                            for j in nodes if j!="Station_0")==1)
    m.addConstr(gp.quicksum(x[i,"Station_0"] 
                            for i in nodes if i!="Station_0")==1)

    if len(stations_14)>1:
        u = m.addVars(stations_14, lb=0, ub=len(stations_14))
        for i in stations_14:
            for j in stations_14:
                if i!=j:
                    m.addConstr(u[i]-u[j]+len(stations_14)*x[i,j] <= len(stations_14)-1)

    m.optimize()
    if m.status==GRB.OPTIMAL:
        return m.ObjVal
    else:
        return 1e9

###############################################################################
def main():
    # 0) Basic setup
    T=20
    S=10
    scenario_count = T*S
    cost_over  = 0.09
    cost_under = 0.13
    random_seed = 1234

    # 1) Read data
    costs_url = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-3/costs.csv"
    df_costs  = read_costs_csv(costs_url)

    rand_url = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-3/randomness.csv"
    df_rand  = read_randomness_csv(rand_url)

    # station_1..14
    station_list = df_rand.index.tolist()
    p_need = {}
    mean_dem = {}
    std_dem  = {}
    for st in station_list:
        p_need[st]   = df_rand.loc[st,"Probability"]
        mean_dem[st] = df_rand.loc[st,"Mean_Demand"]
        std_dem[st]  = df_rand.loc[st,"Std_Dev_Demand"]

    # 2) Generate the 200 scenario demands once, so every approach uses same scenarios
    random.seed(random_seed)
    np.random.seed(random_seed)

    # We'll store for each scenario: demands[s][station] = how much it needs
    scenario_demands = []
    all_stations_14 = [f"Station_{i}" for i in range(1,15)]
    for s in range(scenario_count):
        daily = {}
        for st in all_stations_14:
            if random.random()< p_need[st]:
                d = np.random.normal(mean_dem[st], std_dem[st])
                if d<0: d=0
                daily[st] = d
            else:
                daily[st] = 0.0
        scenario_demands.append(daily)

    # 3) Part (f) => Wait & See cost => solve daily MIP for each scenario
    #    Then compute WS = average of those scenario costs
    total_ws = 0.0
    for s in range(scenario_count):
        cost_s = solve_daily_mip(df_costs, scenario_demands[s], cost_over, cost_under)
        total_ws += cost_s
    WS = total_ws / scenario_count

    # Suppose we also have the "SP cost" from part (e). If we need to solve it now, we'd do
    # a big scenario-based model. For brevity, let's just define it from a prior solution:
    SP_cost = 164.88  # example from part (e)

    EVPI = WS - SP_cost

    print(f"=== PART (f) RESULTS ===")
    print(f"Wait-and-See (WS) = {WS:.2f}")
    print(f"SP cost (from part e) = {SP_cost:.2f}")
    print(f"EVPI = WS - SP_cost = {EVPI:.2f}")

    # 4) Part (g) => Mean Value solution => capacity = sum of means, route visits all stations
    #    Then simulate => EEV, and compute VSS = EEV - SP_cost
    K_EV = sum(mean_dem[st] for st in station_list)  # sum of means
    route_cost_all_stations = solve_tsp_all_stations(df_costs)

    # Now simulate the same 200 scenarios with that approach
    # daily cost = route_cost_all_stations + mismatch cost
    total_ev_sol = 0.0
    for s in range(scenario_count):
        daily_dem = sum(scenario_demands[s].values())  # total actual
        if daily_dem> K_EV:
            mismatch = cost_under*(daily_dem - K_EV)
        else:
            mismatch = cost_over*(K_EV - daily_dem)
        cost_day = route_cost_all_stations + mismatch
        total_ev_sol += cost_day
    EEV = total_ev_sol / scenario_count

    VSS = EEV - SP_cost

    print(f"\n=== PART (g) RESULTS ===")
    print(f"EV solution capacity (sum of means) = {K_EV:.2f}")
    print(f"Route cost (visit all stns)        = {route_cost_all_stations:.2f}")
    print(f"EEV = {EEV:.2f}")
    print(f"SP_cost = {SP_cost:.2f}")
    print(f"VSS = EEV - SP_cost = {VSS:.2f}")

if __name__=="__main__":
    main()
