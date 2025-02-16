import sympy

# 1) Define symbolic variables for the prices
p1, p2 = sympy.symbols('p1 p2', real=True)

# 2) From Table 2 (Part a):
#    Week 1: Intercept=1000, OwnPriceCoeff=-5
#    Week 2: Intercept=950,  OwnPriceCoeff=-4.5
I1, slope1 = 1000, -5
I2, slope2 = 950,  -4.5

# 3) Define the revenue expressions for each week
R1 = p1 * (I1 + slope1 * p1)  # revenue week 1
R2 = p2 * (I2 + slope2 * p2)  # revenue week 2

# 4) Total revenue to maximize (no cross terms)
R_total = R1 + R2

# 5) Take partial derivatives (gradient) wrt p1 and p2
dR_dp1 = sympy.diff(R_total, p1)
dR_dp2 = sympy.diff(R_total, p2)

# 6) Solve the system {dR_dp1=0, dR_dp2=0} for an interior optimum
solutions = sympy.solve(
    [sympy.Eq(dR_dp1, 0), sympy.Eq(dR_dp2, 0)],
    [p1, p2],
    dict=True
)

# In this problem, there should be a single solution
p1_unconstrained = solutions[0][p1]
p2_unconstrained = solutions[0][p2]

# 7) Enforce nonnegativity: if either solution is negative, set price = 0
p1_star = p1_unconstrained if p1_unconstrained >= 0 else 0
p2_star = p2_unconstrained if p2_unconstrained >= 0 else 0

# 8) Print result
print(f"Optimal p1 = {p1_star:.4f}")
print(f"Optimal p2 = {p2_star:.4f}")

# 9) (Optional) Evaluate the total revenue at these prices
R_opt = R_total.subs({p1:p1_star, p2:p2_star})
print(f"Maximum Revenue = {R_opt:.4f}")
