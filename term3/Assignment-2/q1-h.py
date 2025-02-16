import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ 1) Load Data ------------------
url = "https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-2/price_response.csv"
df = pd.read_csv(url)
weeks = sorted(df['Week'].unique())               # e.g. [1,2,...,17]
products = df['Product'].unique()                 # e.g. ["TechFit Smartwatch","PowerSound Earbuds"]

# Dictionary param[(prod, w)] = (Intercept, Own, Cross)
param = {}
for _, row in df.iterrows():
    w = int(row['Week'])
    prod = row['Product']
    Intercept = float(row['Intercept'])
    Own = float(row['Own_Price_Coefficient'])
    Cross = float(row['Cross_Price_Coefficient'])
    param[(prod, w)] = (Intercept, Own, Cross)

# ------------------ 2) Build Gurobi Model ------------------
m = gp.Model("Unconstrained_Dynamic_Pricing")

# Create decision variables for each product-week: p >= 0
p = {}
for prod in products:
    for w in weeks:
        p[(prod, w)] = m.addVar(lb=0.0, name=f"p_{prod[:6]}_w{w}")

# Sum up the revenue over all (prod, week)
obj = gp.QuadExpr()
for w in weeks:
    I_T, O_T, C_T = param[("TechFit Smartwatch", w)]
    I_E, O_E, C_E = param[("PowerSound Earbuds", w)]
    # TechFit portion for week w
    obj += p[("TechFit Smartwatch", w)] * (
        I_T
        + O_T * p[("TechFit Smartwatch", w)]
        + C_T * p[("PowerSound Earbuds", w)]
    )
    # Earbuds portion for week w
    obj += p[("PowerSound Earbuds", w)] * (
        I_E
        + O_E * p[("PowerSound Earbuds", w)]
        + C_E * p[("TechFit Smartwatch", w)]
    )

m.setObjective(obj, GRB.MAXIMIZE)

# Must enable non-convex if there's bilinear terms
m.setParam('NonConvex', 2)

# ------------------ 3) Optimize ------------------
m.optimize()

# ------------------ 4) Plot the resulting weekly prices ------------------
if m.status == GRB.OPTIMAL:
    # Extract solution prices
    tech_prices = [p[("TechFit Smartwatch", w)].X for w in weeks]
    ear_prices  = [p[("PowerSound Earbuds",   w)].X for w in weeks]
    
    # Print the optimal revenue
    print(f"Unconstrained Optimal Revenue = {m.objVal:,.2f}")
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(weeks, tech_prices, 'o-', label="TechFit Smartwatch")
    plt.plot(weeks, ear_prices,  's-', label="PowerSound Earbuds")
    plt.title("Unconstrained Dynamic Pricing (All 17 Weeks)")
    plt.xlabel("Week")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("No optimal solution found. Status:", m.status)


# ========== Model 2: Single (Static) Price per Product over 17 Weeks ==========

m2 = gp.Model("Static_Pricing")

# 2a) Two decision vars: pT >= 0, pE >= 0
pT = m2.addVar(lb=0.0, name="price_TechFit")
pE = m2.addVar(lb=0.0, name="price_Earbuds")

m2.update()

# 2b) Objective: sum_{w=1..17} [ R_w(pT, pE) ]
obj2 = gp.QuadExpr()
for w in weeks:
    I_T, O_T, C_T = param[("TechFit Smartwatch", w)]
    I_E, O_E, C_E = param[("PowerSound Earbuds", w)]
    tech_rev = pT * (I_T + O_T*pT + C_T*pE)
    earb_rev = pE * (I_E + O_E*pE + C_E*pT)
    obj2 += (tech_rev + earb_rev)

m2.setObjective(obj2, GRB.MAXIMIZE)

# 2c) Solve
m2.setParam('NonConvex', 2)
m2.optimize()

# 2d) Report
if m2.status == GRB.OPTIMAL:
    static_revenue = m2.objVal
    print(f"\n--- Fully Static Pricing (1 price across all 17 wks) ---")
    print(f"Optimal Revenue = {static_revenue:,.2f}")
    print(f"TechFit Price = {pT.X:,.2f}")
    print(f"Earbuds Price = {pE.X:,.2f}")
else:
    print("No optimal solution found for static model.")