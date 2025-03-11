import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def check_redundancy():
    # 1) Read Data
    url_welders = (
        "https://raw.githubusercontent.com/"
        "EthanRosehart/schulich_data_science/"
        "refs/heads/main/term3/Mid-term/welders_data.csv"
    )
    welders_df = pd.read_csv(url_welders)

    n_welders = len(welders_df)  # Should be 144
    welder_ids = welders_df["Welder_ID"].to_list()
    safety = welders_df["Safety_Rating"].to_list()
    speed = welders_df["Speed_Rating"].to_list()
    smaw = welders_df["SMAW_Proficient"].to_list()
    gmaw = welders_df["GMAW_Proficient"].to_list()
    fcaw = welders_df["FCAW_Proficient"].to_list()
    gtaw = welders_df["GTAW_Proficient"].to_list()
    experience = welders_df["Experience_10_Years"].to_list()  # 0/1

    # Build sets for constraints
    set_smaw = [i for i in range(n_welders) if smaw[i] == 1]
    set_gmaw = [i for i in range(n_welders) if gmaw[i] == 1]
    set_fcaw = [i for i in range(n_welders) if fcaw[i] == 1]
    set_gtaw = [i for i in range(n_welders) if gtaw[i] == 1]
    set_smaw_gmaw = [i for i in range(n_welders) if smaw[i] == 1 and gmaw[i] == 1]
    set_experience = [i for i in range(n_welders) if experience[i] == 1]

    # speed=1 & safety=3 or 4
    set_speed1_safety34 = [i for i in range(n_welders)
                           if abs(speed[i]-1)<1e-9 and safety[i] in [3,4]]

    set_70_100 = [i for i in range(n_welders) if 70 <= welder_ids[i] <= 100]
    set_101_130 = [i for i in range(n_welders) if 101 <= welder_ids[i] <= 130]

    # Constraints to test
    constraints_to_test = [
        "HireExactly16",
        "SMAW_GMAW_50pct",
        "AtLeast2_SMAW",
        "AtLeast2_GMAW",
        "AtLeast2_FCAW",
        "AtLeast2_GTAW",
        "30pct_Experience",
        "SafetyMinAvg",
        "SpeedMinAvg",
        "AtLeast3_slowButSafe",
        "RangeConstraint"
    ]

    def build_base_model(exclude=None):
        """
        Build and return the Gurobi model for the baseline,
        optionally excluding a single constraint named 'exclude'.
        """
        model = gp.Model("Welders_RedundancyCheck")
        x = model.addVars(n_welders, vtype=GRB.BINARY, name="x")

        # Objective: maximize sum of speeds
        model.setObjective(gp.quicksum(speed[i]*x[i] for i in range(n_welders)), GRB.MAXIMIZE)

        # We'll define each constraint but skip the 'exclude' one
        def add_constr(const_name, constr_logic):
            """Only adds the constraint if const_name != exclude."""
            if const_name != exclude:
                # direct operator-based constraint
                model.addConstr(constr_logic, name=const_name)

        # 1) sum_i x_i = 16
        add_constr("HireExactly16",
                   gp.quicksum(x[i] for i in range(n_welders)) == 16)

        # 2) At least half in SMAW & GMAW => >=8
        add_constr("SMAW_GMAW_50pct",
                   gp.quicksum(x[i] for i in set_smaw_gmaw) >= 8)

        # 3) Each technique >=2
        add_constr("AtLeast2_SMAW", gp.quicksum(x[i] for i in set_smaw) >= 2)
        add_constr("AtLeast2_GMAW", gp.quicksum(x[i] for i in set_gmaw) >= 2)
        add_constr("AtLeast2_FCAW", gp.quicksum(x[i] for i in set_fcaw) >= 2)
        add_constr("AtLeast2_GTAW", gp.quicksum(x[i] for i in set_gtaw) >= 2)

        # 4) At least 5 with >10 yrs
        add_constr("30pct_Experience",
                   gp.quicksum(x[i] for i in set_experience) >= 5)

        # 5) sum of safety >=3.9 *16 =>62.4
        add_constr("SafetyMinAvg",
                   gp.quicksum(safety[i]*x[i] for i in range(n_welders)) >= 62.4)

        # 6) sum of speed >=3.1 *16 =>49.6
        add_constr("SpeedMinAvg",
                   gp.quicksum(speed[i]*x[i] for i in range(n_welders)) >= 49.6)

        # 7) At least 3 with speed=1.0 & safety in {3,4}
        add_constr("AtLeast3_slowButSafe",
                   gp.quicksum(x[i] for i in set_speed1_safety34) >= 3)

        # 8) set_70_100 >=1 +2* set_101_130
        add_constr("RangeConstraint",
                   gp.quicksum(x[i] for i in set_70_100)
                   >= 1 + 2*gp.quicksum(x[i] for i in set_101_130))

        return model, x

    # Build baseline model (all constraints)
    model_base, x_base = build_base_model(exclude=None)
    model_base.optimize()
    if model_base.status == GRB.OPTIMAL:
        base_obj = model_base.objVal
        print(f"\nBase model objective = {base_obj:.4f}")
    else:
        base_obj = None
        print(f"Base model not optimal. Status code = {model_base.status}")

    # Check each constraint for redundancy
    for c_name in constraints_to_test:
        model_test, x_test = build_base_model(exclude=c_name)
        model_test.optimize()
        if model_test.status == GRB.OPTIMAL:
            test_obj = model_test.objVal
            if base_obj is not None:
                if abs(test_obj - base_obj) < 1e-6:
                    print(f"Constraint '{c_name}': objective unchanged => POSSIBLY REDUNDANT")
                elif test_obj > base_obj + 1e-6:
                    print(f"Constraint '{c_name}': objective increased from {base_obj:.4f} to {test_obj:.4f} => definitely NOT redundant.")
                else:
                    print(f"Constraint '{c_name}': objective decreased from {base_obj:.4f} to {test_obj:.4f} => not redundant (active).")
            else:
                print(f"(Base was not feasible/optimal?), can't compare.")
        else:
            print(f"Constraint '{c_name}': Model infeasible or not optimal => definitely NOT redundant.")


if __name__ == "__main__":
    check_redundancy()
