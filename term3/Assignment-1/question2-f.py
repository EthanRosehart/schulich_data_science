#!/usr/bin/env python3

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def main():
    # --------------------------------------------------
    # 1. Load Data & Build Model
    # --------------------------------------------------
    df = pd.read_csv("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/updated_gym_data.csv")  # <-- Replace with actual path/filename
    exercises = df.index.tolist()

    # We'll assume 'Hypertrophy Rating' is the objective coefficient
    HR = df['Hypertrophy Rating'].to_dict()

    # Build the model as usual
    model = gp.Model("RP_Strength")

    # Decision variables: x[i] in [0, 0.05]
    x = model.addVars(exercises, vtype=GRB.CONTINUOUS, lb=0, ub=0.05, name="x")

    # Objective: Maximize sum(x[i]*HR[i])
    model.setObjective(
        gp.quicksum(x[i] * HR[i] for i in exercises),
        sense=GRB.MAXIMIZE
    )

    # Constraint: sum of x[i] = 1
    model.addConstr(gp.quicksum(x[i] for i in exercises) == 1.0, name="TotalProgram")

    # (SFR constraint, etc. - not fully shown here)
    # model.addConstr( ... )
    # ... any other constraints you have ...

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective = {model.ObjVal:.4f}")
    else:
        print(f"Model not optimal/infeasible. Status = {model.status}")
        return

    # --------------------------------------------------
    # 2. Identify "Barbell Back Squats" in the solution
    # --------------------------------------------------
    # We'll locate it by name in your DataFrame
    # e.g. df['Exercise'] == "Barbell Back Squat"
    # This should return the row index.
    squat_rows = df.index[df['Exercise'] == "Barbell back squat"].tolist()
    if not squat_rows:
        print("No row found for Barbell Back Squat in data!")
        return

    squat_idx = squat_rows[0]  # if there's exactly 1 match

    # Get the variable object from Gurobi
    squat_var = x[squat_idx]

    # Current squat HR
    current_hr = HR[squat_idx]

    # Check if the solver included any portion of squats
    if squat_var.X > 1e-6:
        # It's already in the solution
        print(f"\nBarbell Back Squat is used with proportion = {squat_var.X:.4f}.")
        print("Because it's already included, we don't need to raise its HR further.\n")
    else:
        # It's not in the solution; let's check the Reduced Cost or sensitivity info
        print("\nBarbell Back Squat is NOT used in the solution (x=0).")

        # Approach 1: Reduced Cost
        rc = squat_var.RC  # for a max problem, if rc < 0 => need HR to increase by at least |rc|
        print(f"  Reduced Cost = {rc:.4f}")

        # For a *maximization* problem:
        # - If 'rc' is negative, you must raise the objective coefficient (HR) by at least -rc to make the variable enter.
        # - If 'rc' is zero or positive, the variable might be indifferent or constrained by something else.
        required_increase = 0.0
        if rc < 0:
            required_increase = -rc
            new_hr_needed = current_hr + required_increase
            print(f"  => Must increase Barbell Back Squat HR by ~{required_increase:.4f} ")
            print(f"     (from {current_hr:.3f} up to ~{new_hr_needed:.3f}) to include it in the solution.")
        else:
            print("  => The reduced cost is not negative, so it's either borderline or constrained by another factor.")

        # Approach 2: Sensitivity Analysis
        # Gurobi also provides SAObjLow/SAObjUp if the variable is linear in the objective:
        if model.isMIP == 0:  # sensitivity bounds only for (continuous) LP, not MIP
            sa_low = squat_var.SAObjLow
            sa_up  = squat_var.SAObjUp
            print(f"  SAObjLow = {sa_low:.4f}, SAObjUp = {sa_up:.4f}")
            # If 'current_hr < sa_low', you'd need HR >= sa_low to make the variable feasible in solution.
            if current_hr < sa_low:
                needed = sa_low - current_hr
                print(f"  => Need to raise HR from {current_hr:.3f} to at least {sa_low:.3f} (~+{needed:.3f}).")
        else:
            print("  (Exact LP sensitivity info not available for MIPs)")

if __name__ == "__main__":
    main()
