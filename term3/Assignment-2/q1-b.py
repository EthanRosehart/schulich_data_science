import sympy

# Single price p for both weeks
p = sympy.Symbol('p', real=True, nonnegative=True)

# Intercepts and slopes (week 1, week 2)
I1, slope1 = 1000, -5
I2, slope2 = 950, -4.5

# Total revenue for single p:
R = p*(I1 + slope1*p) + p*(I2 + slope2*p)

# Derivative wrt p
dR_dp = sympy.diff(R, p)

# Solve for stationary point
stationary_solution = sympy.solve(sympy.Eq(dR_dp, 0), p)
p_unconstrained = stationary_solution[0]

# Enforce p >= 0
p_star = p_unconstrained if p_unconstrained >= 0 else 0

print(f"Optimal single price: {p_star:.2f}")
