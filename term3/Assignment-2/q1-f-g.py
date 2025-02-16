import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# =========== 1) Load Data ===========
url = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/price_response.csv"
df = pd.read_csv(url)

# =========== 2) Prepare sets and parameters ===========
weeks = sorted(df['Week'].unique())               # [1..17]
products = df['Product'].unique()                # ["TechFit Smartwatch", "PowerSound Earbuds"]

# For quick lookup: demand parameters param[(prod,week)] = (Intercept, OwnCoeff, CrossCoeff)
param = {}
for _, row in df.iterrows():
    w = int(row['Week'])
    prod = row['Product']
    Intercept = float(row['Intercept'])
    Own = float(row['Own_Price_Coefficient'])
    Cross = float(row['Cross_Price_Coefficient'])
    param[(prod, w)] = (Intercept, Own, Cross)

# =========== 3) Build Gurobi Model ===========
m = gp.Model("DynamicPricing_17Weeks")

# Decision variables: p[prod, week] >= 0
# We'll store them in a dict p[(prod,week)]
p = {}
for prod in products:
    for w in weeks:
        p[(prod, w)] = m.addVar(lb=0.0, name=f"price_{prod[:6]}_w{w}")

m.update()

# =========== 4) Objective Function (Quadratic) ===========
#  Revenue = SUM_over_weeks_and_products [ p[prod,w] * (Intercept + Own*p[prod,w] + Cross*p[other,w]) ]
# We'll build this up term by term using Gurobi's .addTerms or a Pythonic sum of gp.QuadExpr
obj = gp.QuadExpr()
for w in weeks:
    # For TechFit:
    I_T, O_T, C_T = param[("TechFit Smartwatch", w)]
    # For Earbuds:
    I_E, O_E, C_E = param[("PowerSound Earbuds", w)]
    # TechFit revenue part:
    obj += p[("TechFit Smartwatch", w)] * (
              I_T
              + O_T * p[("TechFit Smartwatch", w)]
              + C_T * p[("PowerSound Earbuds", w)]
           )
    # Earbuds revenue part:
    obj += p[("PowerSound Earbuds", w)] * (
              I_E
              + O_E * p[("PowerSound Earbuds", w)]
              + C_E * p[("TechFit Smartwatch", w)]
           )

m.setObjective(obj, GRB.MAXIMIZE)

# =========== 5) Price Constraints from Table 1 ===========

#   "1-4: Static pricing for each product throughout this period."
#   => For each product f, p[f,1] = p[f,2] = p[f,3] = p[f,4].
# We'll do that by equality constraints p[f,1] - p[f,2] = 0, etc.
for prod in products:
    # all 1..4 must be same
    m.addConstr(p[(prod, 1)] == p[(prod, 2)])
    m.addConstr(p[(prod, 2)] == p[(prod, 3)])
    m.addConstr(p[(prod, 3)] == p[(prod, 4)])

#   "5-8: Static pricing, at least $10 lower than weeks 1-4"
# For each product f, p[f,5]=p[f,6]=p[f,7]=p[f,8], and p[f,5] <= p[f,1] - 10
for prod in products:
    m.addConstr(p[(prod, 5)] == p[(prod, 6)])
    m.addConstr(p[(prod, 6)] == p[(prod, 7)])
    m.addConstr(p[(prod, 7)] == p[(prod, 8)])
    # at least $10 lower
    # p[f,5] + 10 <= p[f,1]  =>  p[f,1] - p[f,5] >= 10
    m.addConstr(p[(prod, 1)] - p[(prod, 5)] >= 10)

#   "9-11: Static pricing, at least $20 higher than weeks 1-4"
for prod in products:
    m.addConstr(p[(prod, 9)] == p[(prod, 10)])
    m.addConstr(p[(prod, 10)] == p[(prod, 11)])
    # p[f,9] >= p[f,1]+20  => p[f,9] - p[f,1] >= 20
    m.addConstr(p[(prod, 9)] - p[(prod, 1)] >= 20)

#   "12: Price should be the lowest of any week, by at least $5."
# => p[f,12] <= p[f,w] - 5  for all w !=12
# => p[f,w] - p[f,12] >= 5
for prod in products:
    for w in weeks:
        if w != 12:
            m.addConstr(p[(prod, w)] - p[(prod, 12)] >= 5)

#   "13-15: Static pricing for each product, with prices between those in Weeks 1-4 and Weeks 5-8"
# => p[f,13] = p[f,14] = p[f,15]
# => p[f,5] <= p[f,13] <= p[f,1]   (since 5-8 is lower, 1-4 is higher)
for prod in products:
    m.addConstr(p[(prod, 13)] == p[(prod, 14)])
    m.addConstr(p[(prod, 14)] == p[(prod, 15)])
    m.addConstr(p[(prod, 13)] >= p[(prod, 5)])
    m.addConstr(p[(prod, 13)] <= p[(prod, 1)])

#   "16: Prices should be at least $4 higher than in Black Friday (week12)
#        and at least $6 lower than any other week (except 12)."
# => p[f,16] >= p[f,12] + 4
# => For w !=12,16,  p[f,16] <= p[f,w] - 6  => p[f,w] - p[f,16] >= 6
for prod in products:
    m.addConstr(p[(prod, 16)] >= p[(prod, 12)] + 4)
    for w in weeks:
        if w not in [12, 16]:
            m.addConstr(p[(prod, w)] - p[(prod, 16)] >= 6)

#   "17: Prices should be the highest of any week, by at least $15."
# => p[f,17] >= p[f,w] + 15 for w !=17
# => p[f,17] - p[f,w] >= 15
for prod in products:
    for w in weeks:
        if w != 17:
            m.addConstr(p[(prod, 17)] - p[(prod, w)] >= 15)

# =========== 6) Solve ===========
m.setParam('NonConvex', 2)  # Because of p[i] * p[j] cross terms
m.optimize()

# =========== 7) Print Results ===========
if m.status == GRB.OPTIMAL:
    print(f"Optimal Revenue = {m.objVal:,.2f}\n")
    for prod in products:
        print(f"--- {prod} ---")
        for w in weeks:
            val = p[(prod, w)].X
            print(f"Week {w}: price = {val:,.2f}")
        print()
else:
    print("No optimal solution found. Status:", m.status)

import matplotlib.pyplot as plt

# 1) Extract the solution prices from your Gurobi model
#    (this code block assumes you already ran 'm.optimize()' successfully)
techfit_prices = [p[("TechFit Smartwatch", w)].X for w in weeks]
earbuds_prices = [p[("PowerSound Earbuds", w)].X for w in weeks]

# 2) Plot
plt.figure(figsize=(8, 5))  # optional size

plt.plot(weeks, techfit_prices, marker='o', label="TechFit Smartwatch")
plt.plot(weeks, earbuds_prices, marker='s', label="PowerSound Earbuds")

# 3) Labeling & Legend
plt.xlabel("Week")
plt.ylabel("Price")
plt.title("Optimal Price Trajectories Over 17 Weeks")
plt.legend()
plt.grid(True)

# 4) Display
plt.show()