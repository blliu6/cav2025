from functools import reduce
from itertools import product
import sympy as sp
from SumOfSquares import SOSProblem
from utils.Config import CegisConfig
from benchmarks.Examplers import Zone


class SOS:
    def __init__(self, config: CegisConfig, poly_list):
        self.config = config
        self.ex = config.example
        self.n = config.example.n
        self.var_count = 0
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(self.n)])
        self.poly_list = poly_list

    def verify_positive(self, expr, con, deg=2):
        x = self.x
        prob = SOSProblem()
        const = []
        for c in con:
            P, par, terms = self.polynomial(deg)
            const.append(prob.add_sos_constraint(P, x))
            expr = expr - c * P
        expr = sp.expand(expr)
        const.append(prob.add_sos_constraint(expr, x))
        try:
            prob.solve(solver='mosek')
            print('--------------------------------')
            for i, item in enumerate(const):
                if i != len(const) - 1:
                    print('constraint:', con[i])
                    print('Multiplier decomposition results:', sum(item.get_sos_decomp()))
                else:
                    print('Total decomposition results:')
                    print(sum(item.get_sos_decomp()))
            print('--------------------------------')
            return True
        except:
            return False

    def verify_positive_multiplier(self, A, B, con, deg=2, R_deg=2):
        x = self.x
        prob = SOSProblem()
        expr = A
        const = []
        for c in con:
            P, par, terms = self.polynomial(deg)
            const.append(prob.add_sos_constraint(P, x))
            expr = expr - c * P

        R, par, terms = self.polynomial(R_deg)
        par_s = [prob.sym_to_var(item) for item in par]
        expr = expr - R * B
        expr = sp.expand(expr)
        const.append(prob.add_sos_constraint(expr, x))
        try:
            prob.solve(solver='mosek')
            print('--------------------------------')
            for i, item in enumerate(const):
                if i != len(const) - 1:
                    print('constraint:', con[i])
                    print('Multiplier decomposition results:', sum(item.get_sos_decomp()))
                else:
                    value = [item.value for item in par_s]
                    w = sum([x * y for x, y in zip(value, terms)])
                    print('Multiplier:', w)
                    print('Total decomposition results:')
                    print(sum(item.get_sos_decomp()))
            print('--------------------------------')
            return True, w
        except:
            return False, None

    def verify_all(self):
        b1, b2, bm1, bm2, rm1, rm2 = self.poly_list
        deg = self.config.DEG
        x = self.x
        state = [True] * 8
        ################################
        # first
        state[0] = self.verify_positive(b1, self.get_con(self.ex.I), deg=deg[0])
        if not state[0]:
            print('The condition 1 is not satisfied.')
        ################################
        # second
        expr = sum([sp.diff(b1, x[i]) * self.ex.f1[i](x) for i in range(self.n)])
        state[1], R = self.verify_positive_multiplier(expr, b1, self.get_con(self.ex.l1), deg=deg[1])
        # expr = expr - bm1 * b1
        # state[1] = self.verify_positive(expr, self.get_con(self.ex.l1), deg=deg[1])
        if not state[1]:
            print('The condition 2 is not satisfied.')
        ################################
        # third
        expr = sum([sp.diff(b2, x[i]) * self.ex.f2[i](x) for i in range(self.n)])
        state[2], R = self.verify_positive_multiplier(expr, b2, self.get_con(self.ex.l2), deg=deg[2])
        # expr = expr - bm2 * b2
        # state[2] = self.verify_positive(expr, self.get_con(self.ex.l2), deg=deg[2])
        if not state[2]:
            print('The condition 3 is not satisfied.')
        ################################
        # fourth
        b2_fun = sp.lambdify(x, b2)
        x_ = [self.ex.r1[i](x) for i in range(self.n)]
        bl2 = b2_fun(*x_)
        state[3], R = self.verify_positive_multiplier(bl2, b1, self.get_con(self.ex.g1), deg=deg[3])
        # expr = bl2 - rm1 * b1
        # state[3] = self.verify_positive(expr, self.get_con(self.ex.g1), deg=deg[3])
        if not state[3]:
            print('The condition 4 is not satisfied.')
        ################################
        # fifth
        b1_fun = sp.lambdify(x, b1)
        x_ = [self.ex.r2[i](x) for i in range(self.n)]
        bl1 = b1_fun(*x_)
        state[4], R = self.verify_positive_multiplier(bl1, b2, self.get_con(self.ex.g2), deg=deg[4])
        # expr = bl1 - rm2 * b2
        # state[4] = self.verify_positive(expr, self.get_con(self.ex.g2), deg=deg[4])
        if not state[4]:
            print('The condition 5 is not satisfied.')
        ################################
        # sixth
        state[5] = self.verify_positive(rm1, self.get_con(self.ex.g1), deg=deg[5])
        if not state[5]:
            print('The condition 6 is not satisfied.')
        ################################
        # seventh
        state[6] = self.verify_positive(rm2, self.get_con(self.ex.g2), deg=deg[6])
        if not state[6]:
            print('The condition 7 is not satisfied.')
        ################################
        # eighth
        state[7] = self.verify_positive(-b2, self.get_con(self.ex.U), deg=deg[7])
        if not state[7]:
            print('The condition 8 is not satisfied.')

        result = True
        for e in state:
            result = result and e
        # print(result, state)
        return result, state

    def verify_continuous(self):
        b1, bm1 = self.poly_list
        deg = self.config.DEG_continuous
        x = self.x
        state = [True] * 3
        ################################
        # init
        state[0] = self.verify_positive(b1, self.get_con(self.ex.I), deg=deg[0])
        if not state[0]:
            print('The init condition is not satisfied.')
        else:
            print('The init condition is satisfied.')
        ################################
        # Lie
        expr = sum([sp.diff(b1, x[i]) * self.ex.f1[i](x) for i in range(self.n)])
        # expr = expr - bm1 * b1
        # state[1] = self.verify_positive(expr, self.get_con(self.ex.l1), deg=deg[1])
        state[1], R = self.verify_positive_multiplier(expr, b1, self.get_con(self.ex.l1), deg=deg[1], R_deg=deg[2])
        if not state[1]:
            print('The lie condition is not satisfied.')
        else:
            print('The lie condition is satisfied.')
        ################################
        # unsafe
        state[2] = self.verify_positive(-b1, self.get_con(self.ex.U), deg=deg[3])
        if not state[2]:
            print('The unsafe condition is not satisfied.')
        else:
            print('The unsafe condition is satisfied.')

        from SMT.z3_verifyer import smt_verify

        # print('The result of smt:', smt_verify(self.ex, b1, sp.expand(expr - R * b1)))
        result = True
        for e in state:
            result = result and e
        return result, state

    def get_con(self, zone: Zone):
        x = self.x
        if zone.verify_zone is not None:
            return [e(x) for e in zone.verify_zone]
        if zone.shape == 'ball':
            poly = zone.r
            for i in range(self.n):
                poly = poly - (x[i] - zone.center[i]) ** 2
            return [poly]
        elif zone.shape == 'box':
            poly = []
            for i in range(self.n):
                poly.append((x[i] - zone.low[i]) * (zone.up[i] - x[i]))
            return poly

    def polynomial(self, deg=2):  # Generating polynomials of degree n-ary deg.
        if deg == 2:
            parameters = []
            terms = []
            poly = 0
            parameters.append(sp.symbols('parameter' + str(self.var_count)))
            self.var_count += 1
            poly += parameters[-1]
            terms.append(1)
            for i in range(self.n):
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(self.x[i])
                poly += parameters[-1] * terms[-1]
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(self.x[i] ** 2)
                poly += parameters[-1] * terms[-1]
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    parameters.append(sp.symbols('parameter' + str(self.var_count)))
                    self.var_count += 1
                    terms.append(self.x[i] * self.x[j])
                    poly += parameters[-1] * terms[-1]
            return poly, parameters, terms
        else:
            parameters = []
            terms = []
            exponents = list(product(range(deg + 1), repeat=self.n))  # Generate all possible combinations of indices.
            exponents = [e for e in exponents if sum(e) <= deg]  # Remove items with a count greater than deg.
            poly = 0
            for e in exponents:  # Generate all items.
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(reduce(lambda a, b: a * b, [self.x[i] ** exp for i, exp in enumerate(e)]))
                poly += parameters[-1] * terms[-1]
            return poly, parameters, terms


if __name__ == '__main__':
    from benchmarks.Examplers import get_example_by_name
    import sympy as sp

    ex = get_example_by_name('C2')
    bc = sp.sympify('-1.84087207961163e-5*x1 - 0.00723456573535096*x2 - 0.00221299352733997')
    obj = [-bc, '']
    sos = SOS(CegisConfig(**{'example': ex, 'DEG_continuous': [2, 2, 2]}), obj)
    print(sos.verify_continuous())
