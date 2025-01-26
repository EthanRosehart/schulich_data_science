import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def main():
    # ----------------------------------------------------------------
    # 1. Load Data
    # ----------------------------------------------------------------
    df = pd.read_csv("https://raw.githubusercontent.com/EthanRosehart/schulich_data_science/refs/heads/main/term3/Assignment-1/updated_gym_data.csv")  # <-- Replace with actual path/filename
    # Expected columns:
    #   'Exercise', 'Category', 'BodyPart', 'Equipment', 'Difficulty',
    #   'Stimulus-to-Fatigue', 'Expected Time', 'Hypertrophy Rating'

    # Let's create an index for each exercise
    exercises = df.index.tolist()  # 0..2636 if there are 2637 rows
    n = len(exercises)             # should be 2637

    # Extract the needed data from each row into Python dictionaries or lists
    # Example: SFR[i], HR[i], Category[i], BodyPart[i], etc.
    SFR       = df['Stimulus-to-Fatigue'].to_dict()   # SFR[i] in [0..1]
    HR        = df['Hypertrophy Rating'].to_dict()    # HR[i] in [0..1]
    body_part = df['BodyPart'].to_dict()              # e.g. "Chest", "Biceps", ...
    equip     = df['Equipment'].to_dict()             # e.g. "Barbell", "Machine", ...
    cat       = df['Category'].to_dict()              # e.g. "Strongman", "Powerlifting" ...
    diff      = df['Difficulty'].to_dict()            # e.g. "Beginner", "Intermediate", "Advanced"

    # ----------------------------------------------------------------
    # 2. Create a Gurobi Model
    # ----------------------------------------------------------------
    model = gp.Model("RP_Strength_Hypertrophy")

    # Decision variables: x[i] = proportion of program allocated to exercise i
    # We'll allow 0 <= x[i] <= 0.05 as per constraint #1
    x = model.addVars(exercises, vtype=GRB.CONTINUOUS, name="x", lb=0.0, ub=0.05)

    # ----------------------------------------------------------------
    # 3. Objective: Maximize total hypertrophy rating
    #    i.e. sum_{i} (x[i] * HR[i])
    # ----------------------------------------------------------------
    model.setObjective(
        gp.quicksum(x[i] * HR[i] for i in exercises),
        sense=GRB.MAXIMIZE
    )

    # ----------------------------------------------------------------
    # 4. Constraints
    # ----------------------------------------------------------------

    # A) Sum of all proportions = 1 (the entire program)
    #    (This is implied if we want the total to be 100% of the program.)
    model.addConstr(
        gp.quicksum(x[i] for i in exercises) == 1.0,
        name="TotalProgram"
    )

    # B) Minimum proportions for body parts:
    #    - default is 2.5%, except:
    #       * Traps, Neck, Forearms: 0.5%
    #       * Abdominals: 4%
    # We'll define a helper to sum up x[i] for each bodypart
    bodypart_to_min = {
        'Traps': 0.005, 'Neck': 0.005, 'Forearms': 0.005,
        'Abdominals': 0.04
        # all else: 0.025
    }
    # We'll gather all unique body parts from the data
    unique_bodyparts = df['BodyPart'].unique().tolist()

    for bp in unique_bodyparts:
        # sum of x[i] for exercises with that body part
        bodypart_sum = gp.quicksum(x[i] for i in exercises if body_part[i] == bp)
        # Determine min proportion
        if bp in bodypart_to_min:
            min_prop = bodypart_to_min[bp]
        else:
            min_prop = 0.025  # default
        model.addConstr(bodypart_sum >= min_prop, name=f"BodyPartMin_{bp}")

    # C) Leg muscles >= 2.6 * (all upper body)
    #    Let's define sets:
    leg_parts = ['Adductors','Abductors','Calves','Glutes','Hamstrings','Quadriceps']
    upper_parts = ['Chest','Back','Shoulders','Biceps','Triceps','...']  
    # adapt the list to your data structure

    sum_legs = gp.quicksum(x[i] for i in exercises if body_part[i] in leg_parts)
    sum_upper = gp.quicksum(x[i] for i in exercises if body_part[i] in upper_parts)
    model.addConstr(sum_legs >= 2.6 * sum_upper, name="LegVsUpper")

    # D) Biceps proportion = Triceps proportion
    #    Chest proportion = All Back proportion
    sum_biceps = gp.quicksum(x[i] for i in exercises if body_part[i] == "Biceps")
    sum_triceps = gp.quicksum(x[i] for i in exercises if body_part[i] == "Triceps")
    model.addConstr(sum_biceps == sum_triceps, name="BiTriEqual")

    # For "All Back", let's assume body_part = "Back" includes upper/mid/lower
    sum_chest = gp.quicksum(x[i] for i in exercises if body_part[i] == "Chest")
    sum_back = gp.quicksum(x[i] for i in exercises if "Back" in body_part[i])
    model.addConstr(sum_chest == sum_back, name="ChestBackEqual")

    # E) Overall Stimulus-to-Fatigue (SFR) ratio <= 0.55
    #    sum(x[i] * SFR[i]) / sum(x[i]) <= 0.55
    #    but sum(x[i]) = 1, so it's just sum(x[i] * SFR[i]) <= 0.55
    model.addConstr(
        gp.quicksum(x[i] * SFR[i] for i in exercises) <= 0.55,
        name="SFRConstraint"
    )

    # F) Difficulty proportions:
    #    - Beginner >= 1.4 * Intermediate
    #    - Intermediate >= 1.1 * Advanced
    sum_beg = gp.quicksum(x[i] for i in exercises if diff[i] == "Beginner")
    sum_int = gp.quicksum(x[i] for i in exercises if diff[i] == "Intermediate")
    sum_adv = gp.quicksum(x[i] for i in exercises if diff[i] == "Advanced")

    model.addConstr(sum_beg >= 1.4 * sum_int, name="BeginnerVsIntermediate")
    model.addConstr(sum_int >= 1.1 * sum_adv, name="IntermediateVsAdvanced")

    # G) Category constraints: 
    #    - Strongman < 8%
    #    - Powerlifting > 9%
    #    - Olympic Weightlifting > 10%
    sum_strongman = gp.quicksum(x[i] for i in exercises if cat[i] == "Strongman")
    sum_powerlift = gp.quicksum(x[i] for i in exercises if cat[i] == "Powerlifting")
    sum_olympic   = gp.quicksum(x[i] for i in exercises if cat[i] == "Olympic Weightlifting")

    model.addConstr(sum_strongman <= 0.08, name="StrongmanMax")
    model.addConstr(sum_powerlift >= 0.09, name="PowerliftMin")
    model.addConstr(sum_olympic   >= 0.10, name="OlympicMin")

    # H) Equipment usage > 60% for {Barbells, Dumbbells, Machines, Cables, E-Z Curl bar, Bands}
    #    So we sum all x[i] with those equipment types, must be >= 0.60
    eq_allowed = ["Barbell","Dumbbell","Machine","Cable","EZ Curl","Band"]
    eq_sum = gp.quicksum(x[i] for i in exercises if equip[i] in eq_allowed)
    model.addConstr(eq_sum >= 0.60, name="EquipMin")

    # ----------------------------------------------------------------
    # 5. Solve
    # ----------------------------------------------------------------
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal total hypertrophy rating = {model.ObjVal:.4f}")
    else:
        print(f"Model did not solve to optimality. Status code = {model.status}")

if __name__ == "__main__":
    main()