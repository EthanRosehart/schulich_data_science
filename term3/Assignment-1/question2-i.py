import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def main():
    # -----------------------------------------
    # 1) Build PRIMAL model as usual
    # -----------------------------------------
    df = pd.read_csv("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/updated_gym_data.csv")
    exercises = df.index.tolist()

    model = gp.Model("Primal_RP_Strength")

    # For demonstration, assume primal is a continuous LP
    # Decision vars: x[i] in [0, 0.05]
    x = model.addVars(exercises, lb=0, ub=0.05, vtype=GRB.CONTINUOUS, name="x")

    # Set objective, constraints, etc.
    # (Put your constraints #1, #2, #8, or all constraints if you like)
    # e.g. sum(x[i]) == 1, body-part minimums, equipment >= 60%, etc.
    # ...
    # model.addConstr( sum(x[i] for i in exercises) == 1, name="TotalProgram" )
    # etc.

    model.setObjective( 
        gp.quicksum( x[i] * df.loc[i, 'Hypertrophy Rating'] for i in exercises ), 
        sense=GRB.MAXIMIZE
    )

    # Solve the primal
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"Primal not optimal. Status = {model.status}")
        return

    # -----------------------------------------
    # 2) Print Primal Solution
    # -----------------------------------------
    print(f"\n--- PRIMAL Solution (x) ---")
    for v in model.getVars():
        if v.X > 1e-6:
            print(f"{v.VarName} = {v.X:.4f}")

    print(f"\nPrimal Objective Value = {model.ObjVal:.4f}\n")

    # -----------------------------------------
    # 3) Extract Dual Variables from Gurobi
    # -----------------------------------------
    # For each linear constraint, Gurobi provides a shadow price (Pi)
    print("--- DUAL Solution (shadow prices Pi for each constraint) ---")
    for c in model.getConstrs():
        print(f"{c.ConstrName}: Pi={c.Pi:.4f}  (Slack={c.Slack:.4f})")

    # Reduced costs for each variable (RC)
    # In a max problem, if x[i] = 0 and RC < 0 => objective coefficient would need to increase to bring x[i] in
    print("\n--- Reduced Costs (RC) for each variable) ---")
    for v in model.getVars():
        print(f"{v.VarName}: RC={v.RC:.4f}")

    # -----------------------------------------
    # 4) Confirm Dual = Primal (theoretically)
    # -----------------------------------------
    # If the model is in "standard form" (max with <= constraints) + x>=0,
    # the sum of (RHS_j * Pi_j) = primal ObjVal under strong duality.
    #
    # But if you have >= or = constraints, or upper-bounded vars, it complicates direct interpretation.
    # Still, Gurobi's solution respects strong duality for continuous LP.

if __name__ == "__main__":
    main()
