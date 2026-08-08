from fractions import Fraction
import json
from flint import arb, arb_mat, ctx
ctx.prec = 160

def A(q):
    if isinstance(q, Fraction): return arb(q.numerator)/arb(q.denominator)
    if isinstance(q, int): return arb(q)
    return arb(q)

def mat(rows): return arb_mat([[A(x) for x in row] for row in rows])
def eye(n): return arb_mat([[A(1 if i==j else 0) for j in range(n)] for i in range(n)])
def zeros(r,c): return [[A(0) for _ in range(c)] for __ in range(r)]
def scale(M,c): return arb_mat([[M[i,j]*c for j in range(M.ncols())] for i in range(M.nrows())])
def tr(M): return sum((M[i,i] for i in range(M.nrows())), A(0))
def max_abs_entry(M):
    vals=[abs(M[i,j]) for i in range(M.nrows()) for j in range(M.ncols())]
    out=vals[0]
    for v in vals[1:]:
        if v > out: out=v
    return out

def assert_zero_mat(M,label):
    bad=[]
    for i in range(M.nrows()):
        for j in range(M.ncols()):
            if not M[i,j].contains(0): bad.append((i,j,str(M[i,j])))
    if bad: raise AssertionError(f"{label}: entries miss zero: {bad[:5]}")

def assert_zero(x,label):
    if not x.contains(0): raise AssertionError(f"{label}: {x}")

def assert_pos(x,label):
    if not (x > 0): raise AssertionError(f"{label}: not rigorously positive: {x}")

def lyapunov_forced(A3, Q3, s):
    # Solve X' = A X + X A^T + Q, X(0)=0 by a 10x10 augmented matrix exponential.
    n=3; N=10
    aug=zeros(N,N)
    def idx(i,j): return 3*i+j
    for i in range(n):
        for j in range(n):
            row=idx(i,j)
            for k in range(n):
                aug[row][idx(k,j)] += A3[i,k]
                aug[row][idx(i,k)] += A3[j,k]
            aug[row][9] = Q3[i,j]
    Aug=arb_mat(aug)
    E=scale(Aug,s).exp()
    X=arb_mat([[E[idx(i,j),9] for j in range(3)] for i in range(3)])
    return X

def sympart(A3): return scale(A3+A3.transpose(), A(Fraction(1,2)))

def is_zero_matrix_exactish(M):
    return all(M[i,j].contains(0) and str(M[i,j]).strip() in ('0','0.0') for i in range(M.nrows()) for j in range(M.ncols()))

nu=A(Fraction(2,11))
I=eye(3)
cases={
 'zero': [[0,0,0],[0,0,0],[0,0,0]],
 'pure_rotation': [[0,-3,0],[3,0,0],[0,0,0]],
 'simple_shear': [[0,4,0],[0,0,0],[0,0,0]],
 'strain_plus_rotation': [[2,-5,1],[5,-1,0],[0,1,-1]],
 'nonnormal_mixed': [[1,3,-2],[0,-2,4],[1,0,1]],
 'strong_nonnormal': [[7,20,-3],[-2,-5,11],[4,-1,-2]],
}
ss=[Fraction(1,10000),Fraction(1,100),Fraction(1,3),Fraction(1),Fraction(2)]
results=[]

for name,rows in cases.items():
    A3=mat(rows)
    assert_zero(tr(A3),f"trace {name}")
    S=sympart(A3)
    strain_zero = all(S[i,j].contains(0) and rows[i][j] + rows[j][i] == 0 for i in range(3) for j in range(3))
    for sf in ss:
        s=A(sf)
        C=lyapunov_forced(A3, scale(I,2*nu), s)
        M=lyapunov_forced(scale(A3,-1), I, s)
        F=scale(A3,s).exp()
        # incompressible deformation
        assert_zero(F.det()-1,f"det F=1 {name} s={sf}")
        # independent Gramian identity
        gram_res=C-scale(F*M*F.transpose(),2*nu)
        assert_zero_mat(gram_res,f"C=2nu F M F^T {name} s={sf}")
        # precision identity
        prec=F.transpose()*C.inv()*F - scale(M.inv(),1/(2*nu))
        assert_zero_mat(prec,f"precision {name} s={sf}")
        # no-collapse determinant
        gap=C.det()-(2*nu*s)**3
        if strain_zero:
            assert_zero(gap,f"no-collapse equality {name} s={sf}")
        else:
            assert_pos(gap,f"no-collapse strict {name} s={sf}")
        # anisotropy AM-GM
        Ci=C.inv(); trinv=tr(Ci)
        agap=C.det()*(trinv**3)-27
        if strain_zero:
            assert_zero(agap,f"anisotropy equality {name} s={sf}")
        else:
            assert_pos(agap,f"anisotropy strict {name} s={sf}")
        results.append({
          'case':name,'s':str(sf),'strain_zero':strain_zero,
          'detC_gap':str(gap),'anisotropy_gap':str(agap),
          'gram_residual_max':str(max_abs_entry(gram_res)),
          'precision_residual_max':str(max_abs_entry(prec)),
        })
        print(f"PASS {name:20s} s={str(sf):>8s} detC-gap={gap} anisotropy-gap={agap}")

# Pure rotation should reproduce pure Brownian covariance exactly at each sample.
R=mat(cases['pure_rotation'])
for sf in ss:
    s=A(sf); C=lyapunov_forced(R,scale(I,2*nu),s)
    assert_zero_mat(C-scale(I,2*nu*s),f"rotation covariance {sf}")
print('PASS pure rotation leaves covariance exactly isotropic (within Arb enclosure).')

with open('general-affine-results.json','w') as f:
    json.dump({'precision_bits':ctx.prec,'nu':'2/11','cases':results},f,indent=2)
with open('general-affine-summary.md','w') as f:
    f.write('# General affine Arb verification\n\n')
    f.write(f'- Arb precision: **{ctx.prec} bits**\n')
    f.write(f'- Tested matrices: **{len(cases)}** (including shear, rotation, strongly non-normal)\n')
    f.write(f'- Time samples per matrix: **{len(ss)}**\n')
    f.write('- Certified: det(exp(sA)) = 1 for trace-free A.\n')
    f.write('- Certified: C = 2 nu F M F^T using independently evolved forward/backward Gramians.\n')
    f.write('- Certified: F^T C^-1 F = (2 nu)^-1 M^-1.\n')
    f.write('- Certified: det(C) >= (2 nu s)^3; equality for pure rotation/zero strain samples.\n')
    f.write('- Certified: det(C)(tr C^-1)^3 >= 27; equality for isotropic covariance samples.\n')
