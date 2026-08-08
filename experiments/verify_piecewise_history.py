from fractions import Fraction
import json
from flint import arb, arb_mat, ctx
ctx.prec=160

def A(q):
    if isinstance(q,Fraction): return arb(q.numerator)/arb(q.denominator)
    if isinstance(q,int): return arb(q)
    return arb(q)
def mat(rows): return arb_mat([[A(x) for x in r] for r in rows])
def eye(n): return arb_mat([[A(1 if i==j else 0) for j in range(n)] for i in range(n)])
def zmat(n): return arb_mat([[A(0) for _ in range(n)] for __ in range(n)])
def scale(M,c): return arb_mat([[M[i,j]*c for j in range(M.ncols())] for i in range(M.nrows())])
def tr(M): return sum((M[i,i] for i in range(M.nrows())),A(0))
def assert_zero(x,msg):
    if not x.contains(0): raise AssertionError(f'{msg}: {x}')
def assert_pos(x,msg):
    if not (x>0): raise AssertionError(f'{msg}: {x}')
def assert_zero_mat(M,msg):
    for i in range(M.nrows()):
        for j in range(M.ncols()):
            if not M[i,j].contains(0): raise AssertionError(f'{msg}[{i},{j}]={M[i,j]}')

def lyap(A3,Q3,s):
    aug=[[A(0) for _ in range(10)] for __ in range(10)]
    def idx(i,j): return 3*i+j
    for i in range(3):
        for j in range(3):
            row=idx(i,j)
            for k in range(3):
                aug[row][idx(k,j)] += A3[i,k]
                aug[row][idx(i,k)] += A3[j,k]
            aug[row][9]=Q3[i,j]
    Aug=arb_mat(aug)
    Es=arb_mat([[Aug[i,j]*s for j in range(10)] for i in range(10)]).exp()
    return arb_mat([[Es[idx(i,j),9] for j in range(3)] for i in range(3)])

I=eye(3); nu=A(Fraction(3,17))
Rz=[[0,-3,0],[3,0,0],[0,0,0]]
Rx=[[0,0,0],[0,0,-2],[0,2,0]]
S1=[[3,0,0],[0,-1,0],[0,0,-2]]
S2=[[-2,1,0],[1,1,0],[0,0,1]]
Sh=[[0,5,0],[0,0,0],[0,0,0]]
N1=[[2,-4,1],[3,-1,2],[-2,1,-1]]
N2=[[-1,2,5],[0,3,-2],[-4,1,-2]]

histories={
 'rotation_only':[(Rz,Fraction(1,7)),(Rx,Fraction(1,5)),(Rz,Fraction(2,9)),(Rx,Fraction(1,4))],
 'rotating_strain':[(S1,Fraction(1,8)),(Rz,Fraction(1,11)),(S2,Fraction(1,9)),(Rx,Fraction(1,10)),(Sh,Fraction(1,13))],
 'nonnormal_switching':[(N1,Fraction(1,20)),(N2,Fraction(1,17)),(Sh,Fraction(1,19)),(S1,Fraction(1,23)),(N2,Fraction(1,29))],
 'strong_long_history':[(S1,Fraction(1,3)),(Sh,Fraction(1,5)),(Rz,Fraction(1,4)),(N1,Fraction(1,7)),(S2,Fraction(2,9)),(Rx,Fraction(1,6))],
}
results=[]
for hname,segments in histories.items():
    F=I; C=zmat(3); M=zmat(3); elapsed=A(0)
    all_skew=True
    for step,(rows,dtf) in enumerate(segments,1):
        Am=mat(rows); dt=A(dtf); assert_zero(tr(Am),f'trace {hname}/{step}')
        # classify skew exactly from integer matrices
        if any(rows[i][j]+rows[j][i] != 0 for i in range(3) for j in range(3)): all_skew=False
        E=arb_mat([[Am[i,j]*dt for j in range(3)] for i in range(3)]).exp()
        Cseg=lyap(Am,scale(I,2*nu),dt)
        Mseg=lyap(scale(Am,-1),I,dt)
        Finv=F.inv()
        C=E*C*E.transpose()+Cseg
        M=M+Finv*Mseg*Finv.transpose()
        F=E*F
        elapsed += dt
        assert_zero(F.det()-1,f'detF {hname}/{step}')
        gram=C-scale(F*M*F.transpose(),2*nu)
        assert_zero_mat(gram,f'gram identity {hname}/{step}')
        prec=F.transpose()*C.inv()*F-scale(M.inv(),1/(2*nu))
        assert_zero_mat(prec,f'precision {hname}/{step}')
        gap=C.det()-(2*nu*elapsed)**3
        if all_skew: assert_zero(gap,f'rotation equality {hname}/{step}')
        else: assert_pos(gap,f'no-collapse {hname}/{step}')
        Ci=C.inv(); agap=C.det()*(tr(Ci)**3)-27
        if all_skew: assert_zero(agap,f'anisotropy equality {hname}/{step}')
        else: assert_pos(agap,f'anisotropy {hname}/{step}')
        print(f'PASS {hname:20s} step={step} t={elapsed} det-gap={gap} anisotropy-gap={agap}')
        results.append({'history':hname,'step':step,'elapsed':str(elapsed),'det_gap':str(gap),'anisotropy_gap':str(agap)})

with open('piecewise-history-results.json','w') as f: json.dump({'precision_bits':ctx.prec,'results':results},f,indent=2)
with open('piecewise-history-summary.md','w') as f:
    f.write('# Piecewise noncommuting deformation histories\n\n')
    f.write(f'- Arb precision: **{ctx.prec} bits**\n')
    f.write(f'- Histories: **{len(histories)}**; total segment checks: **{len(results)}**\n')
    f.write('- Every segment matrix is trace-free; histories include changing rotation axes, shear, strain, and non-normal switching.\n')
    f.write('- Certified after every segment: det F = 1, C = 2 nu F M F^T, and F^T C^-1 F = (2nu)^-1 M^-1.\n')
    f.write('- Certified after every strained history step: det C > (2 nu t)^3 and anisotropy defect > 0.\n')
    f.write('- Rotation-only history stays on the exact isotropic equality branch within Arb enclosure.\n')
