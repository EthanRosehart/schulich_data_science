#!/usr/bin/env python3

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def main():
    # --------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------
    df = pd.read_csv("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/updated_gym_data.csv")

    # We expect columns like:
    #   Exercise, Category, BodyPart, Equipment, Difficulty,
    #   Stimulus-to-Fatigue, Expected Time, Hypertrophy Rating

    # Index set for exercises
    exercises = df.index.tolist()

    # Create dictionaries from columns
    HR = df['Hypertrophy Rating'].to_dict()
    body_part = df['BodyPart'].to_dict()
    equip = df['Equipment'].to_dict()

    # --------------------------------------------------
    # 2. Build Gurobi Model
    # --------------------------------------------------
    model = gp.Model("RP_Strength_MiniConstraints")

    # Decision vars: x[i] in [0, 0.05]
    x = model.addVars(exercises, vtype=GRB.CONTINUOUS, lb=0, ub=0.05, name="x")

    # Objective: Maximize sum(x[i] * HR[i])
    model.setObjective(
        gp.quicksum(x[i] * HR[i] for i in exercises),
        sense=GRB.MAXIMIZE
    )

    # --------------------------------------------------
    # 3. Constraints
    # --------------------------------------------------

    # (A) Sum of x[i] = 1
    model.addConstr(
        gp.quicksum(x[i] for i in exercises) == 1.0,
        name="TotalProgram"
    )

    # (B) Body-part minimums
    # Default = 2.5% (0.025)
    # Traps, Neck, Forearms => 0.5% (0.005)
    # Abdominals => 4% (0.04)
    special_min = {
        'Traps': 0.005,
        'Neck': 0.005,
        'Forearms': 0.005,
        'Abdominals': 0.04
        # everything else => 0.025
    }

    unique_bps = df['BodyPart'].unique()

    for bp_val in unique_bps:
        sum_bp = gp.quicksum(x[i] for i in exercises if body_part[i] == bp_val)
        min_req = special_min.get(bp_val, 0.025)
        model.addConstr(sum_bp >= min_req, name=f"Min_{bp_val}")

    # (C) Equipment usage >= 60% among [Barbell, Dumbbell, Machine, Cable, E-Z Curl bar, Band]
    eq_allowed = ["Barbell", "Dumbbell", "Machine", "Cable", "E-Z Curl bar", "Band"]
    eq_sum = gp.quicksum(x[i] for i in exercises if equip[i] in eq_allowed)
    model.addConstr(eq_sum >= 0.60, name="EquipMin60")

    # --------------------------------------------------
    # 4. Solve Model
    # --------------------------------------------------
    model.optimize()

    # --------------------------------------------------
    # 5. Print Results
    # --------------------------------------------------
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal total hypertrophy rating = {model.ObjVal:.4f}\n")

        # Store solution in df
        df["x"] = df.index.map(lambda i: x[i].X)
        # Filter out zero or near-zero
        chosen = df[df["x"] > 1e-6].copy()
        # Sort in descending order
        chosen.sort_values("x", ascending=False, inplace=True)

        print("Chosen Exercises and Proportions:")
        for idx, row in chosen.iterrows():
            ex_name = row["Exercise"]
            ex_hr   = row["Hypertrophy Rating"]
            ex_x    = row["x"]
            print(f"  {ex_name:40s} x={ex_x:.4f}, HR={ex_hr:.3f}")
    else:
        print(f"Model did not solve optimally (status={model.status}).")

if __name__ == "__main__":
    main()
