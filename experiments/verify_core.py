from fractions import Fraction
import json
import sympy as sp
from flint import arb, ctx

ctx.prec = 160
RESULTS = []

def rec(kind, name, status, detail):
    row = {"kind": kind, "name": name, "status": status, "detail": detail}
    RESULTS.append(row)
    print(f"[{kind}] {name}: {status}\n  {detail}")

def A(x):
    if isinstance(x, Fraction):
        return arb(x.numerator) / arb(x.denominator)
    if isinstance(x, int): return arb(x)
    if isinstance(x, str): return arb(x)
    return arb(x)

def assert_zero_enclosure(x, label):
    if not x.contains(0): raise AssertionError(f"{label}: interval does not contain 0: {x}")

def assert_positive(x, label):
    if not (x > 0): raise AssertionError(f"{label}: interval is not rigorously positive: {x}")

def diag_covariance(lams, nu, s):
    cs, ms, fs = [], [], []
    for q in lams:
        lam = A(q)
        if q == 0:
            c = 2 * nu * s; m = s; f = A(1)
        else:
            f = (lam * s).exp()
            c = nu * ((2 * lam * s).exp() - 1) / lam
            m = (1 - (-2 * lam * s).exp()) / (2 * lam)
        cs.append(c); ms.append(m); fs.append(f)
    return cs, ms, fs

# EXACT 1
xi = sp.Matrix([sp.Rational(1,3), sp.Rational(2,3), sp.Rational(2,3)])
H = sp.Matrix([[2,1,0],[1,-1,1],[0,1,-1]])
I3 = sp.eye(3); Q = xi*xi.T - I3/sp.Integer(3)
frob = lambda M: sp.trace(M.T*M)
lhs = -1 + 2*sp.trace(H.T*Q) - sp.Rational(2,3)*frob(H)
rhs = -sp.Rational(2,3)*frob(H - sp.Rational(3,2)*Q)
assert sp.simplify(lhs-rhs) == 0
rec("EXACT", "tensor-square-completion", "PASS", "identity residual = 0")

# EXACT 2
k = sp.Matrix([1,2,3]); k2 = (k.T*k)[0]
P = I3 - (k*k.T)/k2
assert sp.simplify(P*P-P) == sp.zeros(3)
assert sp.simplify(P.T-P) == sp.zeros(3)
assert sp.simplify(P*k) == sp.zeros(3,1)
omega = sp.Matrix([2,-1,0]); assert (k.T*omega)[0] == 0
u = k.cross(omega)/k2
Shat = (k*u.T + u*k.T)/2
strain2 = sp.simplify(frob(Shat)); omega2 = sp.simplify((omega.T*omega)[0])
assert sp.simplify(strain2 - omega2/2) == 0
rec("EXACT", "Hodge-projector-and-L2-strain", "PASS", f"P^2=P, Pk=0; |S_hat|^2/|omega|^2 = {sp.simplify(strain2/omega2)}")

# EXACT 3
m1,m2,m3 = sp.symbols('m1 m2 m3')
Cross = sp.Matrix([[0,-m3,m2],[m3,0,-m1],[-m2,m1,0]])
Enn = I3/sp.Integer(3)
assert sp.simplify(-Cross*Enn + Enn*Cross) == sp.zeros(3)
rec("EXACT", "constant-vorticity-shell-cancellation", "PASS", "isotropic shell mean strain tensor = 0 exactly")

# EXACT 4
P0 = sp.diag(1,1,0); Q0 = I3-P0
Omega = sp.Matrix([[0,0,1],[0,0,0],[-1,0,0]])
v0 = sp.Matrix([1,2,0]); Pdot = Omega*P0 - P0*Omega; vdot = Omega*v0
assert sp.simplify(Q0*vdot - Pdot*v0) == sp.zeros(3,1)
rec("EXACT", "moving-Hodge-pressure-geometry", "PASS", "Q v_dot = P_dot v exactly in moving-subspace model")

# ARB-RIGOROUS 1
nu_arb = A(Fraction(1,7))
strain_cases = [
    (Fraction(0), Fraction(0), Fraction(0)),
    (Fraction(3), Fraction(-1), Fraction(-2)),
    (Fraction(1), Fraction(1), Fraction(-2)),
    (Fraction(7), Fraction(-3), Fraction(-4)),
    (Fraction(1,4), Fraction(-1,10), Fraction(-3,20)),
    (Fraction(25), Fraction(-12), Fraction(-13)),
]
s_cases = [Fraction(1,1000000), Fraction(1,1000), Fraction(1,10), Fraction(1), Fraction(3)]
min_positive_gap = None; checked = 0
for lams in strain_cases:
    assert sum(lams) == 0
    for sf in s_cases:
        s = A(sf); cs, ms, fs = diag_covariance(lams, nu_arb, s)
        gap = cs[0]*cs[1]*cs[2] - (2*nu_arb*s)**3
        if lams == (Fraction(0), Fraction(0), Fraction(0)):
            assert_zero_enclosure(gap, f"no-collapse equality {lams}, s={sf}")
        else:
            assert_positive(gap, f"no-collapse strict {lams}, s={sf}")
            if min_positive_gap is None or gap < min_positive_gap: min_positive_gap = gap
        checked += 1
rec("ARB-RIGOROUS", "covariance-no-collapse", "PASS", f"{checked} cases; det(C) >= (2 nu s)^3 at {ctx.prec} bits; smallest sampled strict gap={min_positive_gap}")

# ARB-RIGOROUS 2
checked = 0; min_gap = None
for lams in strain_cases:
    for sf in s_cases[1:]:
        s=A(sf); cs, _, _ = diag_covariance(lams, nu_arb, s)
        detC=cs[0]*cs[1]*cs[2]; trinv=1/cs[0]+1/cs[1]+1/cs[2]
        gap=detC*(trinv**3)-27
        if lams == (Fraction(0), Fraction(0), Fraction(0)):
            assert_zero_enclosure(gap, "AMGM equality")
        else:
            assert_positive(gap, f"AMGM strict {lams}, s={sf}")
            if min_gap is None or gap < min_gap: min_gap=gap
        checked += 1
rec("ARB-RIGOROUS", "covariance-anisotropy-defect", "PASS", f"{checked} cases; det(C)*(tr C^-1)^3 >= 27; smallest sampled strict gap={min_gap}")

# ARB-RIGOROUS 3
checked = 0
for lams in strain_cases:
    for sf in s_cases:
        s=A(sf); cs, ms, fs = diag_covariance(lams, nu_arb, s)
        for i in range(3):
            diff = fs[i]*fs[i]/cs[i] - 1/(2*nu_arb*ms[i])
            assert_zero_enclosure(diff, f"precision identity {lams}, s={sf}, i={i}")
            checked += 1
rec("ARB-RIGOROUS", "Cauchy-Fisher-precision-identity", "PASS", f"{checked} component identities enclosed zero: F^T C^-1 F = (2nu)^-1 M^-1")

# NUMERICAL-EVIDENCE
rows=[]
for lamf in [Fraction(3), Fraction(-2), Fraction(1,4)]:
    lam=A(lamf); target=-(lam**4)/45
    for sf in [Fraction(1,100), Fraction(1,1000), Fraction(1,10000), Fraction(1,100000)]:
        s=A(sf); m=(1-(-2*lam*s).exp())/(2*lam)
        cubic=(1/m-(1/s+lam+(lam*lam*s)/3))/(s**3)
        err=cubic-target; rows.append((str(lamf),str(sf),str(cubic),str(err)))
for lamf in [Fraction(3), Fraction(-2), Fraction(1,4)]:
    lam=A(lamf); s=A(Fraction(1,100000)); target=-(lam**4)/45
    m=(1-(-2*lam*s).exp())/(2*lam)
    cubic=(1/m-(1/s+lam+(lam*lam*s)/3))/(s**3)
    assert abs(cubic-target) < A(Fraction(1,10000000))
rec("NUMERICAL-EVIDENCE", "small-s-precision-expansion", "PASS", "Arb point samples approach M^-1 = s^-1 I + S + (s/3)S^2 - (s^3/45)S^4 + ...; evidence only")

print("\n=== SMALL-S TABLE ===")
for r in rows: print("lambda=%s s=%s cubic_residual=%s error=%s" % r)

summary = {"arb_precision_bits": ctx.prec, "counts": {"exact": sum(r['kind']=='EXACT' for r in RESULTS), "arb_rigorous": sum(r['kind']=='ARB-RIGOROUS' for r in RESULTS), "numerical_evidence": sum(r['kind']=='NUMERICAL-EVIDENCE' for r in RESULTS)}, "results": RESULTS}
with open('verification-results.json','w') as f: json.dump(summary,f,indent=2)
with open('verification-summary.md','w') as f:
    f.write('# Verification summary\n\n'); f.write(f'- Arb precision: **{ctx.prec} bits**\n')
    for r in RESULTS: f.write(f"- **{r['kind']} / {r['name']}**: {r['status']} — {r['detail']}\n")
