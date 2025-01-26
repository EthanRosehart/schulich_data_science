"""
RP Strength Model - SFR Sensitivity
-----------------------------------
We solve the same "Region, 3%, 50% constraints" model twice:
  1) With SFR <= 0.55
  2) With SFR <= 0.551
Then we compare the difference in objective (hypertrophy rating).

Note: You must adapt muscle groups, equipment categories, etc. to your data if they differ.
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_model_with_sfr(sfr_limit):
    """
    Builds and solves the RP Strength model with the given 'sfr_limit'.
    Returns the optimal objective value (hypertrophy rating), or None if infeasible.
    """
    # --------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------
    df = pd.read_csv("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/updated_gym_data.csv")  # <-- Replace with actual path/filename

    # Index set for exercises
    exercises = df.index.tolist()
    n = len(exercises)

    # Extract data
    SFR       = df['Stimulus-to-Fatigue'].to_dict()
    HR        = df['Hypertrophy Rating'].to_dict()
    body_part = df['BodyPart'].to_dict()
    equip     = df['Equipment'].to_dict()
    cat       = df['Category'].to_dict()
    diff      = df['Difficulty'].to_dict()

    # Create model
    model = gp.Model("RP_Strength_SFR_Sensitivity")

    # Decision vars: x[i], proportion for each exercise
    x = model.addVars(exercises, vtype=GRB.CONTINUOUS, lb=0.0, ub=0.05, name="x")

    # Objective: Maximize sum(x[i]*HR[i])
    model.setObjective(
        gp.quicksum(x[i] * HR[i] for i in exercises),
        sense=GRB.MAXIMIZE
    )

    # Constraint: sum of all x[i] = 1
    model.addConstr(gp.quicksum(x[i] for i in exercises) == 1.0, name="TotalProgram")

    # --------------------------------------------------
    # Example Additional Constraints
    # (Add your region, 3%, 50% constraints, etc. as needed)
    # --------------------------------------------------

    # E) SFR: sum(x[i]*SFR[i]) <= sfr_limit * sum(x[i]) = sfr_limit * 1
    # Because sum(x[i]) = 1, so sum(x[i]*SFR[i]) <= sfr_limit
    model.addConstr(
        gp.quicksum(x[i]*SFR[i] for i in exercises) <= sfr_limit,
        name="SFRConstraint"
    )

    # ... Here you'd include all your other constraints
    #     - (Region) y[p,c] = 0 if mismatch, etc. (if relevant)
    #     - (3% each plant, 50% center) from your previous code
    #     - (Minimum muscle proportions, etc.)
    #
    # For simplicity here, we just show the SFR logic as an example.

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.ObjVal
    else:
        return None

def main():
    # Solve at SFR limit = 0.55 (base)
    base_sfr = 0.55
    obj_base = solve_model_with_sfr(base_sfr)

    if obj_base is None:
        print(f"Base scenario (SFR=0.55) is infeasible or not optimal.")
        return

    print(f"Base scenario (SFR=0.55) => Objective = {obj_base:.4f}")

    # Solve at SFR limit = 0.551 (relaxed by 0.001)
    relaxed_sfr = 0.551
    obj_relaxed = solve_model_with_sfr(relaxed_sfr)

    if obj_relaxed is None:
        print(f"Relaxed scenario (SFR=0.551) is infeasible or not optimal.")
        return

    print(f"Relaxed scenario (SFR=0.551) => Objective = {obj_relaxed:.4f}")

    # Compare
    improvement = obj_relaxed - obj_base
    print(f"By relaxing SFR from {base_sfr} to {relaxed_sfr}, "
          f"the hypertrophy rating improves by ~{improvement:.4f}.")

if __name__ == "__main__":
    main()