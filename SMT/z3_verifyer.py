import sympy as sp
from z3 import Reals, Solver, sat, unsat
from benchmarks.Examplers import Example, Zone, get_example_by_name


def verify(n, zone: Zone, expr):
    x = Reals(' '.join([f'x{i + 1}' for i in range(n)]))
    solver = Solver()
    if zone.shape == 'box':
        for i, (l, r) in enumerate(zip(zone.low, zone.up)):
            solver.add(x[i] >= l, x[i] <= r)
    else:
        s = zone.r
        for i in range(n):
            s -= (x[i] - zone.center[i]) ** 2
        solver.add(s >= 0)
    x_ = sp.symbols([f'x{i + 1}' for i in range(n)])
    lam = sp.lambdify(x_, expr)
    solver.add(lam(*x) < 0)
    if solver.check() == unsat:
        return True
    else:
        print(solver.model())
        return False


def smt_verify(ex: Example, barrier, db):
    # verify init
    s1 = verify(ex.n, ex.I, barrier)

    # verify unsafe
    s2 = verify(ex.n, ex.U, -barrier)

    # verify domain
    s3 = verify(ex.n, ex.l1, db)

    if s1 and s2 and s3:
        print('验证通过！')
        return True
    else:
        print('验证失败！')
        return False


if __name__ == '__main__':
    bc = get_example_by_name('C1')
    verify(3, bc.I, 0)
