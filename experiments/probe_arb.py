from flint import arb, arb_mat, ctx
ctx.prec = 160
print('ctx.prec =', ctx.prec)
print('arb example =', arb(1) / arb(3))
A = arb_mat([[arb(0), arb(1)], [arb(-1), arb(0)]])
print('arb_mat dir subset:', [x for x in dir(A) if x in {'exp','inv','det','transpose','solve','charpoly'}])
for name in ('exp','inv','det','transpose'):
    obj = getattr(A, name, None)
    print(name, 'available=', obj is not None)
    if callable(obj):
        try:
            out = obj()
            print(name, 'works:', out)
        except Exception as e:
            print(name, 'error:', type(e).__name__, str(e))
