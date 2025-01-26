import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def main():
    # ----------------------------------------------------------------
    # 1. Load Data
    # ----------------------------------------------------------------
    df = pd.read_csv("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/updated_gym_data.csv")
    # Columns: Exercise, Category, BodyPart, Equipment, Difficulty,
    # Stimulus-to-Fatigue, Expected Time, Hypertrophy Rating

    exercises = df.index.tolist()

    # Extract dictionaries
    SFR       = df['Stimulus-to-Fatigue'].to_dict()
    HR        = df['Hypertrophy Rating'].to_dict()
    body_part = df['BodyPart'].to_dict()
    equip     = df['Equipment'].to_dict()
    cat       = df['Category'].to_dict()
    diff      = df['Difficulty'].to_dict()

    # ----------------------------------------------------------------
    # 2. Create Gurobi Model
    # ----------------------------------------------------------------
    model = gp.Model("RP_Strength_Hypertrophy")

    # Decision vars: x[i] in [0, 0.05]
    x = model.addVars(exercises, vtype=GRB.CONTINUOUS, lb=0.0, ub=0.05, name="x")

    # ----------------------------------------------------------------
    # 3. Objective: Maximize sum(x[i]*HR[i])
    # ----------------------------------------------------------------
    model.setObjective(
        gp.quicksum(x[i] * HR[i] for i in exercises),
        sense=GRB.MAXIMIZE
    )

    # ----------------------------------------------------------------
    # 4. Constraints
    # ----------------------------------------------------------------

    # (A) Sum of x[i] = 1
    model.addConstr(
        gp.quicksum(x[i] for i in exercises) == 1.0,
        name="TotalProgram"
    )

    # (B) Body-part minimums
    bodypart_to_min = {
        'Traps': 0.005,      # 0.5%
        'Neck': 0.005,
        'Forearms': 0.005,
        'Abdominals': 0.04   # 4%
        # all else default: 0.025 (2.5%)
    }
    unique_bodyparts = df['BodyPart'].unique().tolist()

    for bp_val in unique_bodyparts:
        sum_bp = gp.quicksum(x[i] for i in exercises if body_part[i] == bp_val)
        min_req = bodypart_to_min.get(bp_val, 0.025)
        model.addConstr(sum_bp >= min_req, name=f"BodyPartMin_{bp_val}")

    # (C) Leg vs. Upper
    leg_parts = ['Adductors','Abductors','Calves','Glutes','Hamstrings','Quadriceps']
    upper_parts = ['Chest','Back','Shoulders','Biceps','Triceps','...']

    sum_legs = gp.quicksum(x[i] for i in exercises if body_part[i] in leg_parts)
    sum_upper = gp.quicksum(x[i] for i in exercises if body_part[i] in upper_parts)
    model.addConstr(sum_legs >= 2.6 * sum_upper, name="LegVsUpper")

    # (D) Biceps=Triceps, Chest=AllBack
    sum_biceps = gp.quicksum(x[i] for i in exercises if body_part[i] == "Biceps")
    sum_triceps = gp.quicksum(x[i] for i in exercises if body_part[i] == "Triceps")
    model.addConstr(sum_biceps == sum_triceps, name="BiTriEqual")

    sum_chest = gp.quicksum(x[i] for i in exercises if body_part[i] == "Chest")
    sum_back  = gp.quicksum(x[i] for i in exercises if "Back" in body_part[i])
    model.addConstr(sum_chest == sum_back, name="ChestBackEqual")

    # (E) SFR <= 0.55
    model.addConstr(
        gp.quicksum(x[i] * SFR[i] for i in exercises) <= 0.55,
        name="SFRConstraint"
    )

    # (F) Difficulty proportions
    sum_beg = gp.quicksum(x[i] for i in exercises if diff[i] == "Beginner")
    sum_int = gp.quicksum(x[i] for i in exercises if diff[i] == "Intermediate")
    sum_adv = gp.quicksum(x[i] for i in exercises if diff[i] == "Advanced")

    model.addConstr(sum_beg >= 1.4 * sum_int, name="BeginnerVsIntermediate")
    model.addConstr(sum_int >= 1.1 * sum_adv, name="IntermediateVsAdvanced")

    # (G) Category constraints
    sum_strongman = gp.quicksum(x[i] for i in exercises if cat[i] == "Strongman")
    sum_powerlift = gp.quicksum(x[i] for i in exercises if cat[i] == "Powerlifting")
    sum_olympic   = gp.quicksum(x[i] for i in exercises if cat[i] == "Olympic Weightlifting")

    model.addConstr(sum_strongman <= 0.08, name="StrongmanMax")
    model.addConstr(sum_powerlift >= 0.09, name="PowerliftMin")
    model.addConstr(sum_olympic   >= 0.10, name="OlympicMin")

    # (H) Equipment usage >= 60%
    eq_allowed = ["Barbell","Dumbbell","Machine","Cable","EZ Curl","Band"]
    eq_sum = gp.quicksum(x[i] for i in exercises if equip[i] in eq_allowed)
    model.addConstr(eq_sum >= 0.60, name="EquipMin")

    # ----------------------------------------------------------------
    # 5. Solve
    # ----------------------------------------------------------------
    model.optimize()

    # ----------------------------------------------------------------
    # 6. Print Results
    # ----------------------------------------------------------------
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal total hypertrophy rating = {model.ObjVal:.4f}\n")
        
        # Convert solution to a column in df
        df["x"] = df.index.map(lambda i: x[i].X)

        # Filter out near-zero
        chosen = df[df["x"] > 1e-6].copy()
        # Sort descending by proportion
        chosen.sort_values(by="x", ascending=False, inplace=True)

        print("Chosen Exercises and Proportions:")
        for idx, row in chosen.iterrows():
            print(f"  {row['Exercise']:<40s} x={row['x']:.4f}, HR={row['Hypertrophy Rating']:.3f}")
    else:
        print(f"Model did not solve to optimality. Status code = {model.status}")

if __name__ == "__main__":
    main()
