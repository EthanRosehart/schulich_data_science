import math

# --- 1) Constants from CSV (Weeks 1 & 2 only) ---
# Week 1, TechFit:
I_T1 = 294.306794322898
O_T1 = -1.4914293088352972
C_T1 = 0.2815132947852186

# Week 1, PowerSound:
I_E1 = 274.7876687446625
O_E1 = -1.5896170769643703
C_E1 = 0.2511102277086097

# Week 2, TechFit:
I_T2 = 268.63944964748674
O_T2 = -1.9230200901712071
C_T2 = 0.15795029058275362

# Week 2, PowerSound:
I_E2 = 248.36638617620133
O_E2 = -1.070302347657427
C_E2 = 0.2616240759128834

# --- 2) Gradient function for R(p_T, p_E) ---
def grad_revenue(pT, pE):
    # partial wrt pT:
    dRd_pT = (
        (I_T1 + 2*O_T1*pT + C_T1*pE) +
        (I_T2 + 2*O_T2*pT + C_T2*pE) +
        pE*(C_E1 + C_E2)
    )
    # partial wrt pE:
    dRd_pE = (
        pT*(C_T1 + C_T2) +
        (I_E1 + 2*O_E1*pE + C_E1*pT) +
        (I_E2 + 2*O_E2*pE + C_E2*pT)
    )
    return dRd_pT, dRd_pE

# --- 3) Gradient Ascent with Projection ---
eta = 0.001          # step size
tol = 1e-6           # stopping criterion
max_iter = 500000    # safety cap on iterations

pT = 0.0  # initialize
pE = 0.0

for iteration in range(max_iter):
    dRd_pT, dRd_pE = grad_revenue(pT, pE)
    
    # Take a step in +gradient direction (because we maximize)
    new_pT = pT + eta*dRd_pT
    new_pE = pE + eta*dRd_pE
    
    # Project onto p >= 0
    new_pT = max(new_pT, 0.0)
    new_pE = max(new_pE, 0.0)
    
    # Check stopping criterion
    dist = math.sqrt((new_pT - pT)**2 + (new_pE - pE)**2)
    if dist < tol:
        pT, pE = new_pT, new_pE
        break
    
    # Otherwise, update and continue
    pT, pE = new_pT, new_pE

print(f"Converged after {iteration+1} iterations.")
print(f"Optimal TechFit price (weeks 1&2) = {pT:.4f}")
print(f"Optimal PowerSound price (weeks 1&2) = {pE:.4f}")