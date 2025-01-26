"""
RP Strength Model (Inspection of Special Muscle Groups)
-------------------------------------------------------
We solve the model with all your constraints:
 - sum(x[i]) = 1
 - x[i] <= 0.05
 - SFR <= 0.55 (example)
 - etc...

Then we specifically inspect how the solver allocates to
Traps, Neck, Forearms, Abdominals. We'll also look at
the top 5-10 highest-rated exercises in those groups
to see if they made it into the solution.
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def build_and_solve(df):
    """
    Builds and solves the model given the DataFrame df.
    Returns (model_status, objective_value, x_solution) where:
      - x_solution is a dict of {exercise_index: proportion}
    """
    # Index
    exercises = df.index.tolist()

    # Extract relevant columns
    SFR = df['Stimulus-to-Fatigue'].to_dict()
    HR  = df['Hypertrophy Rating'].to_dict()

    # Create model
    model = gp.Model("RP_Strength")

    # Decision variables: x[i] in [0, 0.05]
    x = model.addVars(exercises, vtype=GRB.CONTINUOUS, lb=0, ub=0.05, name="x")

    # Objective: Maximize sum(x[i]*HR[i])
    model.setObjective(
        gp.quicksum(x[i] * HR[i] for i in exercises),
        sense=GRB.MAXIMIZE
    )

    # Basic constraint: sum of x[i] = 1
    model.addConstr(
        gp.quicksum(x[i] for i in exercises) == 1.0,
        name="TotalProgram"
    )

    # Example: SFR <= 0.55
    model.addConstr(
        gp.quicksum(x[i]*SFR[i] for i in exercises) <= 0.55,
        name="SFRConstraint"
    )

    # --------------------------------------------------
    # Here you would add your other constraints:
    #  - minimum muscle group proportions
    #  - difficulty proportions
    #  - category constraints
    #  - etc.
    # --------------------------------------------------

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        sol = {i: x[i].X for i in exercises}
        return (model.status, model.ObjVal, sol)
    else:
        return (model.status, None, {})

def main():
    # Load your CSV (adjust path)
    df = pd.read_csv("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/updated_gym_data.csv")  # <-- Replace with actual path/filename

    # Solve the model
    status, obj_val, x_sol = build_and_solve(df)

    if status != GRB.OPTIMAL:
        print(f"Model is infeasible or not optimal (status={status}).")
        return

    print(f"Optimal hypertrophy rating: {obj_val:.4f}\n")

    # Build a DataFrame of the solution: each row is an exercise, with columns:
    #  [ ExerciseName, BodyPart, x, HR, SFR, ... ]
    df['x'] = df.index.map(x_sol)           # proportion chosen by solver
    df['x'] = df['x'].fillna(0.0)           # fill with 0 if not in solution dict
    df['ChosenFlag'] = (df['x'] > 1e-6)     # True if chosen above tiny threshold

    # We want to inspect these body parts
    special_groups = ["Traps", "Neck", "Forearms", "Abdominals"]

    for muscle in special_groups:
        print(f"===== Muscle Group: {muscle} =====")
        # Filter to only exercises that target 'muscle'
        sub_df = df[df['BodyPart'] == muscle].copy()

        if len(sub_df) == 0:
            print(f"  No exercises found for {muscle}\n")
            continue

        # 1) Print the exercises that the solver actually used (x > 0)
        used_ex = sub_df[sub_df['ChosenFlag'] == True]
        if not used_ex.empty:
            used_ex = used_ex.sort_values('x', ascending=False)
            print("  Chosen Exercises (solver allocated > 0):")
            for idx, row in used_ex.iterrows():
                ex_name = row['Exercise']
                ex_x    = row['x']
                ex_hr   = row['Hypertrophy Rating']
                print(f"    {ex_name}: proportion={ex_x:.4f}, HR={ex_hr:.3f}")
        else:
            print("  Solver did NOT allocate any proportion to this group.\n")

        # 2) Show the top N highest-rated exercises in that group,
        #    to see if the solver used them or not
        topN = 5  # you can choose 5, 10, etc.
        top_hr = sub_df.sort_values('Hypertrophy Rating', ascending=False).head(topN)
        print(f"  Top {topN} by Hypertrophy Rating in {muscle}:")
        for idx, row in top_hr.iterrows():
            ex_name = row['Exercise']
            ex_hr   = row['Hypertrophy Rating']
            ex_x    = row['x']
            chosen  = ("YES" if ex_x > 1e-6 else "NO")
            print(f"    {ex_name}: HR={ex_hr:.3f}, x={ex_x:.4f}, Chosen={chosen}")

        print()

if __name__ == "__main__":
    main()