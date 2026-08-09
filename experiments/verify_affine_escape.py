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
    if isinstance(x, sp.Rational):
        return arb(int(x.p)) / arb(int(x.q))
    if isinstance(x, int):
        return arb(x)
    if isinstance(x, str):
        return arb(x)
    return arb(x)

def assert_zero_ball(x, label):
    if not x.contains(0):
        raise AssertionError(f"{label}: interval does not contain 0: {x}")

def assert_positive(x, label):
    if not (x > 0):
        raise AssertionError(f"{label}: interval not rigorously positive: {x}")

# -----------------------------------------------------------------------------
# EXACT GAUSSIAN / AFFINE PROJECTION CALCULUS
# -----------------------------------------------------------------------------
x, y, z, a = sp.symbols('x y z a', positive=True, real=True)
r1, r2, r3 = sp.symbols('r1 r2 r3', real=True)
X = (x, y, z)
R = sp.Matrix([r1, r2, r3])

# Nonlinear, divergence-free polynomial field with affine + quadratic pieces.
u = sp.Matrix([
    2*x + y**2 + y*z,
    -y + z**2 + z*x,
    -z + x**2 + x*y,
])
assert sp.simplify(sum(sp.diff(u[i], X[i]) for i in range(3))) == 0

def lap(expr):
    return sum(sp.diff(expr, q, 2) for q in X)

def heat_poly(expr):
    out = sp.expand(expr)
    cur = sp.expand(expr)
    n = 0
    while True:
        cur = sp.expand(lap(cur))
        n += 1
        if cur == 0:
            break
        out += a**n * cur / sp.factorial(n)
        if n > 8:
            raise RuntimeError("unexpected polynomial heat depth")
    return sp.expand(out)

def gmom(n):
    if n % 2:
        return sp.Integer(0)
    if n == 0:
        return sp.Integer(1)
    return sp.factorial2(n-1) * (2*a)**(n//2)

def gexpect(expr):
    p = sp.Poly(sp.expand(expr), r1, r2, r3)
    out = 0
    for powers, coeff in p.terms():
        out += coeff * gmom(powers[0]) * gmom(powers[1]) * gmom(powers[2])
    return sp.expand(out)

ua = sp.Matrix([heat_poly(ui) for ui in u])
Amat = ua.jacobian(X)
assert sp.simplify(sp.trace(Amat)) == 0

subs_shift = {x:x+r1, y:y+r2, z:z+r3}
u_shift = sp.Matrix([sp.expand(ui.subs(subs_shift, simultaneous=True)) for ui in u])
fluct = sp.expand(u_shift - ua)
N = sp.expand(fluct - Amat*R)

meanN = sp.Matrix([sp.simplify(gexpect(N[i])) for i in range(3)])
assert meanN == sp.zeros(3,1)
for i in range(3):
    for j, rr in enumerate((r1,r2,r3)):
        assert sp.simplify(gexpect(N[i]*rr)) == 0
rec("EXACT", "gaussian-best-affine-orthogonality", "PASS",
    "E[N]=0 and E[N tensor R]=0 exactly for a nonlinear divergence-free polynomial field")

P_u2 = heat_poly(sum(ui**2 for ui in u))
e = sp.expand(sp.Rational(1,2) * (P_u2 - sum(v**2 for v in ua)))
EN2 = sp.expand(sum(gexpect(N[i]**2) for i in range(3)))
frobA = sp.expand(sum(Amat[i,j]**2 for i in range(3) for j in range(3)))
assert sp.simplify(e - a*frobA - sp.Rational(1,2)*EN2) == 0
rec("EXACT", "subscale-energy-pythagoras", "PASS",
    "e_a = a|grad u_a|^2 + (1/2) E|N_a|^2 exactly")

lap_e_over_a = sum(sp.diff(e/a, q, 2) for q in X)
scale_lhs = sp.diff(e/a, a) - lap_e_over_a
scale_rhs = -EN2/(2*a**2)
assert sp.simplify(scale_lhs-scale_rhs) == 0
rec("EXACT", "scale-monotonicity-nonaffinity-sink", "PASS",
    "(d_a-Delta)(e_a/a) = -E|N_a|^2/(2a^2) exactly")

# Tensor stress split.
tau = sp.Matrix(3,3, lambda i,j: sp.expand(gexpect(fluct[i]*fluct[j])))
Rres = sp.Matrix(3,3, lambda i,j: sp.expand(gexpect(N[i]*N[j])))
assert sp.simplify(tau - 2*a*Amat*Amat.T - Rres) == sp.zeros(3)
rec("EXACT", "reynolds-stress-affine-residual-split", "PASS",
    "tau_a = 2a A_a A_a^T + R_a with R_a = E[N_a tensor N_a]")

S = sp.simplify((Amat+Amat.T)/2)
omega = sp.Matrix([
    sp.diff(ua[2],y)-sp.diff(ua[1],z),
    sp.diff(ua[0],z)-sp.diff(ua[2],x),
    sp.diff(ua[1],x)-sp.diff(ua[0],y),
])
frob_contract = lambda M,Nm: sp.expand(sum(M[i,j]*Nm[i,j] for i in range(3) for j in range(3)))
left_cubic = frob_contract(S, Amat*Amat.T)
right_cubic = sp.trace(S**3) - sp.Rational(1,4)*(omega.T*S*omega)[0]
assert sp.simplify(left_cubic-right_cubic) == 0
pi_local = -frob_contract(S,tau)
pi_split = -2*a*sp.trace(S**3) + a*sp.Rational(1,2)*(omega.T*S*omega)[0] - frob_contract(S,Rres)
assert sp.simplify(pi_local-pi_split) == 0
rec("EXACT", "cubic-affine-flux-decomposition", "PASS",
    "S:AA^T = tr(S^3) - (1/4) omega.S.omega and local flux split holds exactly")

# Sampled rigorous positivity of the true non-affinity sink.
sample_points = [
    (Fraction(0),Fraction(0),Fraction(0),Fraction(1,100)),
    (Fraction(1,3),Fraction(-2,5),Fraction(1,7),Fraction(1,10)),
    (Fraction(2),Fraction(-1),Fraction(3,2),Fraction(3,7)),
    (Fraction(-4,3),Fraction(5,4),Fraction(-2,3),Fraction(2)),
]
for xx,yy,zz,aa in sample_points:
    val = sp.factor(EN2.subs({x:sp.Rational(xx.numerator,xx.denominator), y:sp.Rational(yy.numerator,yy.denominator), z:sp.Rational(zz.numerator,zz.denominator), a:sp.Rational(aa.numerator,aa.denominator)}))
    if not isinstance(val, sp.Rational):
        val = sp.Rational(val)
    assert_positive(A(val), f"non-affinity sampled positivity at {(xx,yy,zz,aa)}")
rec("ARB-RIGOROUS", "sampled-nonaffinity-strict-positivity", "PASS",
    f"{len(sample_points)} nonlinear Gaussian clouds have rigorously positive E|N_a|^2 at {ctx.prec} bits")

# -----------------------------------------------------------------------------
# EXACT PERIODIC BETCHOV TEST ON A NONTRIVIAL FOURIER TRIAD
# -----------------------------------------------------------------------------
Iunit = sp.I

def negk(k): return tuple(-q for q in k)

k1=(1,1,0); k2=(1,0,1); k3=(2,1,1)
V1=[(1,-1,0),(0,0,1),(1,-1,2)]
V2=[(1,0,-1),(0,1,0),(1,2,-1)]
V3=[(1,-2,0),(1,0,-2),(0,1,-1)]
PHASES=[
    ('cos','cos','cos'),('cos','cos','sin'),('cos','sin','cos'),('cos','sin','sin'),
    ('sin','cos','cos'),('sin','cos','sin'),('sin','sin','cos'),('sin','sin','sin'),
]

def build_modes_exact(spec):
    v1,v2,v3,kinds=spec
    out={}
    def add(k,vec,amp,kind):
        amp=sp.Rational(amp.numerator,amp.denominator)
        v=sp.Matrix([sp.Rational(q) for q in vec])
        assert sum(sp.Integer(k[j])*v[j] for j in range(3)) == 0
        if kind=='cos': cp=amp*v/2; cm=amp*v/2
        else: cp=-Iunit*amp*v/2; cm=Iunit*amp*v/2
        out[k]=out.get(k,sp.zeros(3,1))+cp
        nk=negk(k); out[nk]=out.get(nk,sp.zeros(3,1))+cm
    add(k1,v1,Fraction(1),kinds[0])
    add(k2,v2,Fraction(3,5),kinds[1])
    add(k3,v3,Fraction(7,6),kinds[2])
    return out

def Ahat_exact(k, uv):
    return sp.Matrix(3,3, lambda i,j: Iunit*sp.Integer(k[j])*uv[i])
def Shat_exact(k, uv):
    G=Ahat_exact(k,uv); return sp.simplify((G+G.T)/2)
def what_exact(k, uv):
    kk=sp.Matrix(k); return sp.simplify(Iunit*kk.cross(uv))

def exact_I(modes_in):
    Sex0={k:Shat_exact(k,v) for k,v in modes_in.items()}
    wex0={k:what_exact(k,v) for k,v in modes_in.items()}
    keys0=list(modes_in); out=0
    for ka in keys0:
        for kb in keys0:
            kc=tuple(-(ka[j]+kb[j]) for j in range(3))
            if kc in modes_in:
                out += (wex0[ka].T*Sex0[kb]*wex0[kc])[0]
    return sp.simplify(out)

chosen_spec=None; Iex=None
for v1 in V1:
    if chosen_spec is not None: break
    for v2 in V2:
        if chosen_spec is not None: break
        for v3 in V3:
            if chosen_spec is not None: break
            for kinds in PHASES:
                spec=(v1,v2,v3,kinds)
                cand=exact_I(build_modes_exact(spec))
                if cand != 0:
                    chosen_spec=spec; Iex=cand; break
if chosen_spec is None:
    raise AssertionError("exact rational search found no nonzero vortex-stretching triad")

modes_exact=build_modes_exact(chosen_spec)
Aex={k:Ahat_exact(k,v) for k,v in modes_exact.items()}
Sex={k:Shat_exact(k,v) for k,v in modes_exact.items()}
wex={k:what_exact(k,v) for k,v in modes_exact.items()}
keys=list(modes_exact)
Iex=0; T3ex=0; SAAex=0
for ka in keys:
    for kb in keys:
        kc=tuple(-(ka[j]+kb[j]) for j in range(3))
        if kc not in modes_exact: continue
        Iex += (wex[ka].T*Sex[kb]*wex[kc])[0]
        T3ex += sp.trace(Sex[ka]*Sex[kb]*Sex[kc])
        SAAex += frob_contract(Sex[ka], Aex[kb]*Aex[kc].T)
Iex=sp.simplify(Iex); T3ex=sp.simplify(T3ex); SAAex=sp.simplify(SAAex)
assert sp.simplify(T3ex + sp.Rational(3,4)*Iex) == 0
assert sp.simplify(SAAex + Iex) == 0
assert Iex != 0
rec("EXACT", "betchov-and-global-affine-work", "PASS",
    f"CI-selected nonzero rational triad: spec={chosen_spec}; exact I={Iex}; Betchov and <S:AA^T>=-<I> hold")
# -----------------------------------------------------------------------------
# ARB COMPLEX-PAIR FOURIER CALCULUS: FILTERED FLUX IDENTITY
# -----------------------------------------------------------------------------
def C(re=0, im=0): return [A(re), A(im)]
def cadd(z1,z2): return [z1[0]+z2[0], z1[1]+z2[1]]
def csub(z1,z2): return [z1[0]-z2[0], z1[1]-z2[1]]
def cmul(z1,z2): return [z1[0]*z2[0]-z1[1]*z2[1], z1[0]*z2[1]+z1[1]*z2[0]]
def cscale(z1,r): return [z1[0]*r, z1[1]*r]
def cimul(z1): return [-z1[1], z1[0]]
def cz(): return C(0,0)
def csum(vals):
    out=cz()
    for v in vals: out=cadd(out,v)
    return out

def vzero(): return [cz(),cz(),cz()]
def vadd(v,w): return [cadd(v[i],w[i]) for i in range(3)]
def vscale(v,r): return [cscale(v[i],r) for i in range(3)]
def vdot(v,w): return csum(cmul(v[i],w[i]) for i in range(3))
def vcross_real(k,v):
    return [
        csub(cscale(v[2],A(k[1])),cscale(v[1],A(k[2]))),
        csub(cscale(v[0],A(k[2])),cscale(v[2],A(k[0]))),
        csub(cscale(v[1],A(k[0])),cscale(v[0],A(k[1]))),
    ]
def mzero(): return [[cz() for _ in range(3)] for _ in range(3)]
def madd(M,Nm): return [[cadd(M[i][j],Nm[i][j]) for j in range(3)] for i in range(3)]
def msub(M,Nm): return [[csub(M[i][j],Nm[i][j]) for j in range(3)] for i in range(3)]
def mscale(M,r): return [[cscale(M[i][j],r) for j in range(3)] for i in range(3)]
def mtrans(M): return [[M[j][i] for j in range(3)] for i in range(3)]
def mmul(M,Nm):
    return [[csum(cmul(M[i][k],Nm[k][j]) for k in range(3)) for j in range(3)] for i in range(3)]
def mvec(M,v): return [csum(cmul(M[i][j],v[j]) for j in range(3)) for i in range(3)]
def mfrob(M,Nm): return csum(cmul(M[i][j],Nm[i][j]) for i in range(3) for j in range(3))
def mtrace(M): return csum(M[i][i] for i in range(3))
def outer(v,w): return [[cmul(v[i],w[j]) for j in range(3)] for i in range(3)]

def grad_hat(k,v):
    M=mzero()
    for i in range(3):
        for j in range(3):
            M[i][j]=cscale(cimul(v[i]),A(k[j]))
    return M

def strain_hat(k,v):
    G=grad_hat(k,v); return mscale(madd(G,mtrans(G)),A(Fraction(1,2)))
def vort_hat(k,v): return [cimul(q) for q in vcross_real(k,v)]
def kadd(k,l): return tuple(k[i]+l[i] for i in range(3))
def ksq(k): return sum(q*q for q in k)

modes={}
def add_mode(k,vec,amp,kind):
    ampA=A(amp); half=A(Fraction(1,2)); vv=[A(q) for q in vec]
    if sum(k[i]*int(vec[i]) for i in range(3)) != 0:
        raise AssertionError("mode not divergence-free")
    if kind=='cos':
        vp=[C(vv[i]*ampA*half,0) for i in range(3)]
        vm=[C(vv[i]*ampA*half,0) for i in range(3)]
    else:
        vp=[C(0,-vv[i]*ampA*half) for i in range(3)]
        vm=[C(0, vv[i]*ampA*half) for i in range(3)]
    modes[k]=vp; modes[negk(k)]=vm

_cv1,_cv2,_cv3,_ckinds=chosen_spec
add_mode(k1,_cv1,Fraction(1),_ckinds[0])
add_mode(k2,_cv2,Fraction(3,5),_ckinds[1])
add_mode(k3,_cv3,Fraction(7,6),_ckinds[2])
def filt_field(field, aa):
    return {k:vscale(v,(-aa*A(ksq(k))).exp()) for k,v in field.items()}

def conv_outer(field1,field2,q):
    out=mzero()
    for k,v in field1.items():
        l=tuple(q[i]-k[i] for i in range(3))
        if l in field2: out=madd(out,outer(v,field2[l]))
    return out

def cubic_quantities(field):
    ks=list(field)
    Gh={k:grad_hat(k,v) for k,v in field.items()}
    Sh={k:strain_hat(k,v) for k,v in field.items()}
    wh={k:vort_hat(k,v) for k,v in field.items()}
    Iv=cz(); T3=cz(); SAA=cz()
    for ka in ks:
        for kb in ks:
            kc=tuple(-(ka[j]+kb[j]) for j in range(3))
            if kc not in field: continue
            Iv=cadd(Iv,vdot(wh[ka],mvec(Sh[kb],wh[kc])))
            T3=cadd(T3,mtrace(mmul(Sh[ka],mmul(Sh[kb],Sh[kc]))))
            SAA=cadd(SAA,mfrob(Sh[ka],mmul(Gh[kb],mtrans(Gh[kc]))))
    return Gh,Sh,wh,Iv,T3,SAA

arb_flux_rows=[]; nonzero_flux=0
for af in [Fraction(1,100),Fraction(1,7),Fraction(1,2),Fraction(3,2)]:
    aa=A(af); ufa=filt_field(modes,aa)
    Gh,Sh,wh,Iv,T3,SAA=cubic_quantities(ufa)
    # Betchov and global affine work for the filtered field.
    rb=cadd(T3,cscale(Iv,A(Fraction(3,4))))
    ra=cadd(SAA,Iv)
    for part,label in [(rb[0],'Betchov-real'),(rb[1],'Betchov-imag'),(ra[0],'SAA-real'),(ra[1],'SAA-imag')]:
        assert_zero_ball(part,f"{label} a={af}")

    tau={}; Rhat={}
    for q in ufa:
        uu=conv_outer(modes,modes,q)
        Pu=mscale(uu,(-aa*A(ksq(q))).exp())
        uaua=conv_outer(ufa,ufa,q)
        tau[q]=msub(Pu,uaua)
        AAq=mzero()
        for k,Gk in Gh.items():
            l=tuple(q[i]-k[i] for i in range(3))
            if l in Gh: AAq=madd(AAq,mmul(Gk,mtrans(Gh[l])))
        Rhat[q]=msub(tau[q],mscale(AAq,2*aa))

    Stau=cz(); SR=cz()
    for q,Sq in Sh.items():
        nq=negk(q)
        if nq in tau: Stau=cadd(Stau,mfrob(Sq,tau[nq]))
        if nq in Rhat: SR=cadd(SR,mfrob(Sq,Rhat[nq]))
    Pi=cscale(Stau,A(-1))
    resid=csub(Pi,csub(cscale(Iv,2*aa),SR))
    assert_zero_ball(resid[0],f"filtered flux real residual a={af}")
    assert_zero_ball(resid[1],f"filtered flux imag residual a={af}")
    assert_zero_ball(Pi[1],f"Pi imaginary a={af}")
    if not Pi[0].contains(0): nonzero_flux += 1
    arb_flux_rows.append((str(af),str(Pi[0]),str(Iv[0]),str(SR[0]),str(resid[0])))

if nonzero_flux == 0:
    raise AssertionError("Fourier test accidentally had zero resolved flux at all scales")
rec("ARB-RIGOROUS", "filtered-flux-affine-residual-identity", "PASS",
    f"{len(arb_flux_rows)} heat scales; Pi=2a I-<S:R> rigorously, with nonzero Pi in {nonzero_flux} cases")

# -----------------------------------------------------------------------------
# ARB-RIGOROUS TRUE-TIME SANITY: EXACT BELTRAMI NAVIER-STOKES SOLUTION
# -----------------------------------------------------------------------------
# Normalize ||u_0||_2^2 = 1 and |k|^2=1. Then K_a(t)=1/2 exp(-2 nu t-2a),
# W(t)=exp(-2 nu t), Pi_a=0. Along a=nu(T-t), verify d_tau A_hor=A_hor-W.
nu=A(Fraction(2,7)); T=A(Fraction(5,2))
bel_rows=[]
for sf in [Fraction(1,100),Fraction(1,10),Fraction(1,2),Fraction(1),Fraction(2)]:
    s=A(sf); t=T-s; aa=nu*s
    q=(-2*nu*t).exp(); em2a=(-2*aa).exp()
    W=q
    Esub=q*(1-em2a)/2
    Ah=Esub/aa
    # g(a)=(1-e^-2a)/(2a), g'(a) exact.
    g=(1-em2a)/(2*aa)
    gp=(2*aa*em2a-(1-em2a))/(2*aa*aa)
    dta=-aa*q*(2*g+gp)  # s*d_t(q*g), with da/dt=-nu
    rhs=Ah-W
    rr=dta-rhs
    assert_zero_ball(rr,f"Beltrami horizon accounting s={sf}")
    assert_positive(W-Ah,f"Beltrami W-A positivity s={sf}")
    bel_rows.append((str(sf),str(Ah),str(W-Ah),str(rr)))
rec("ARB-RIGOROUS", "beltrami-viscous-horizon-accounting", "PASS",
    f"{len(bel_rows)} true Navier-Stokes times; Pi=0 and d_tau A_hor=A_hor-||omega||_2^2 rigorously")

print("\n=== FILTERED FOURIER FLUX TABLE ===")
for row in arb_flux_rows:
    print("a=%s Pi=%s I=%s <S:R>=%s residual=%s" % row)
print("\n=== BELTRAMI HORIZON TABLE ===")
for row in bel_rows:
    print("s=%s A_hor=%s W-A=%s residual=%s" % row)

summary={
    "arb_precision_bits":ctx.prec,
    "counts":{
        "exact":sum(r['kind']=='EXACT' for r in RESULTS),
        "arb_rigorous":sum(r['kind']=='ARB-RIGOROUS' for r in RESULTS),
    },
    "results":RESULTS,
    "fourier_flux_rows":arb_flux_rows,
    "beltrami_rows":bel_rows,
}
with open('affine-escape-results.json','w') as f: json.dump(summary,f,indent=2)
with open('affine-escape-summary.md','w') as f:
    f.write('# Affine Escape verification summary\n\n')
    f.write(f'- Arb precision: **{ctx.prec} bits**\n')
    for r in RESULTS:
        f.write(f"- **{r['kind']} / {r['name']}**: {r['status']} — {r['detail']}\n")
