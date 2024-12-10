import time

import cvxpy as cp
import numpy as np
import sympy as sp
from scipy.optimize import minimize, NonlinearConstraint
import multiprocessing as mp
from multiprocessing import Pool

from benchmarks.Examplers import Zone
from utils.Config import CegisConfig
from verify.counter_examples import get_maximum_volume_ellipsoid


def split_bounds(bounds, n):
    """
    Divide an n-dimensional cuboid into 2^n small cuboids, and output the upper and lower bounds of each small cuboid.

    parameter: bounds: An array of shape (n, 2), representing the upper and lower bounds of each dimension of an
    n-dimensional cuboid.

    return:
        An array with a shape of (2^n, n, 2), representing the upper and lower bounds of the divided 2^n small cuboids.
    """

    if n == bounds.shape[0]:
        return bounds.reshape((-1, *bounds.shape))
    else:
        # Take the middle position of the upper and lower bounds of the current dimension as the split point,
        # and divide the cuboid into two small cuboids on the left and right.
        if n > 5 and np.random.random() > 0.5:
            subbounds = split_bounds(bounds, n + 1)
        else:
            mid = (bounds[n, 0] + bounds[n, 1]) / 2
            left_bounds = bounds.copy()
            left_bounds[n, 1] = mid
            right_bounds = bounds.copy()
            right_bounds[n, 0] = mid
            # Recursively divide the left and right small cuboids.
            left_subbounds = split_bounds(left_bounds, n + 1)
            right_subbounds = split_bounds(right_bounds, n + 1)
            # Merge the upper and lower bounds of the left and right small cuboids into an array.
            subbounds = np.concatenate([left_subbounds, right_subbounds])

        return subbounds


class CounterExampleFinder:
    def __init__(self, config: CegisConfig):
        self.config = config
        self.ex = self.config.example
        self.n = self.ex.n
        self.nums = config.counterexample_nums
        self.ellipsoid = config.counterexamples_ellipsoid
        self.eps = config.eps
        self.lam_con = None

    def filter_point(self, data):
        data_t = np.array(data).T
        vis = np.stack([lam(*data_t) >= 0 for lam in self.lam_con], axis=1)
        vis = np.sum(vis, axis=1)
        data = [item for i, item in enumerate(data) if vis[i] == len(self.lam_con)]
        return data

    def get_circle_center(self, constraints):
        x = sp.symbols([f'x{i + 1}' for i in range(self.n)])
        lam_con = [sp.lambdify(x, item) for item in constraints]
        self.lam_con = lam_con
        self.con = constraints
        cons = [{'type': 'ineq', 'fun': lambda y: item(*y)} for item in lam_con]

        res = minimize(lambda y: 0, np.zeros(self.ex.n), constraints=cons)
        if res.success:
            return True, res.x
        else:
            return False, None

    def get_counterexample(self, zone: Zone, expr):
        constraints = []
        x = sp.symbols([f'x{i + 1}' for i in range(self.n)])
        if zone.shape == 'box':
            for i, (l, r) in enumerate(zip(zone.low, zone.up)):
                constraints.append(sp.expand(-(x[i] - l) * (x[i] - r)))
        elif zone.shape == 'ball':
            poly = zone.r
            for i in range(self.ex.n):
                poly = poly - (x[i] - zone.center[i]) ** 2
            constraints.append(sp.expand(poly))

        constraints.append(-expr)
        state, center = self.get_circle_center(constraints)
        if state:
            if self.ellipsoid:
                return self.filter_point(self.enhance(center))

            state1, counter_points = get_maximum_volume_ellipsoid(center, constraints, counter_nums=self.nums)

            if state1:
                return self.filter_point(counter_points)
            else:
                return self.filter_point(self.enhance(center))
        else:
            return []

    def find_counterexample(self, state, poly_list):
        expr = self.get_expr(poly_list)

        l1, l2, I, U, g1, g2, l1_dot, l2_dot = [], [], [], [], [], [], [], []

        if not state[0]:
            x = self.get_counterexample(self.ex.I, expr[0])
            I.extend(x)

        if not state[7]:
            x = self.get_counterexample(self.ex.U, expr[7])
            U.extend(x)

        if not state[1]:
            x = self.get_counterexample(self.ex.l1, expr[1])
            l1.extend(x)
            l1_dot.extend(self.x2dotx(x, self.ex.f1))

        if not state[2]:
            x = self.get_counterexample(self.ex.l2, expr[2])
            l2.extend(x)
            l2_dot.extend(self.x2dotx(x, self.ex.f2))

        if not state[3]:
            x = self.get_counterexample(self.ex.g1, expr[3])
            g1.extend(x)

        if not state[4]:
            x = self.get_counterexample(self.ex.g2, expr[4])
            g2.extend(x)

        res = (l1, l2, I, U, g1, g2, l1_dot, l2_dot)
        return res

    def find_counterexample_for_continuous(self, state, poly_list):
        expr = self.get_expr_for_continuous(poly_list)

        l1, I, U, l1_dot = [], [], [], []

        if not state[0]:
            x = self.get_counterexample(self.ex.I, expr[0])
            I.extend(x)

        if not state[2]:
            x = self.get_counterexample(self.ex.U, -expr[0])
            U.extend(x)

        if not state[1]:
            if self.config.split and self.ex.l1 == 'box':
                bounds = self.split_zone(self.ex.l1)
            else:
                bounds = [self.ex.l1]
            for e in bounds:
                x = self.get_counterexample(e, expr[1])
                l1.extend(x)
                l1_dot.extend(self.x2dotx(x, self.ex.f1))

        res = (l1, I, U, l1_dot)
        return res

    def enhance(self, x, r=None):
        eps = self.eps if r is None else r
        nums = self.nums
        s = np.random.randn(nums, self.n)
        s = np.array([e / np.sqrt(sum(e ** 2)) * eps * np.random.random() ** (1 / self.n) for e in s])
        result = s + x

        return list(result)

    def split_zone(self, zone: Zone):
        bound = list(zip(zone.low, zone.up))
        bounds = split_bounds(np.array(bound), 0)
        ans = [Zone(shape='box', low=e.T[0], up=e.T[1]) for e in bounds]
        return ans

    def x2dotx(self, X, f):
        X = np.array(X)
        XT = X.T
        res = [func(XT) for func in f]
        return np.stack(res, axis=1)

    def get_expr(self, poly_list):
        b1, b2, bm1, bm2, rm1, rm2 = poly_list

        x = sp.symbols([f'x{i + 1}' for i in range(self.n)])
        expr = sum([sp.diff(b1, x[i]) * self.ex.f1[i](x) for i in range(self.n)])
        expr1 = expr - bm1 * b1

        expr = sum([sp.diff(b2, x[i]) * self.ex.f2[i](x) for i in range(self.n)])
        expr2 = expr - bm2 * b2

        b2_fun = sp.lambdify(x, b2)
        x_ = [self.ex.r1[i](x) for i in range(self.n)]
        bl2 = b2_fun(*x_)
        expr3 = bl2 - rm1 * b1

        b1_fun = sp.lambdify(x, b1)
        x_ = [self.ex.r2[i](x) for i in range(self.n)]
        bl1 = b1_fun(*x_)
        expr4 = bl1 - rm2 * b2

        return [b1, expr1, expr2, expr3, expr4, rm1, rm2, -b2]

    def get_expr_for_continuous(self, poly_list):
        b1, bm1 = poly_list
        x = sp.symbols([f'x{i + 1}' for i in range(self.n)])
        expr = sum([sp.diff(b1, x[i]) * self.ex.f1[i](x) for i in range(self.n)])
        expr = expr - bm1 * b1

        return [b1, expr]


if __name__ == '__main__':
    pass
    # from benchmarks.Examplers import get_example_by_name
    #
    # zone = Zone(shape='box', low=[0, 0], up=[1, 1])
    #
    # ex = get_example_by_name('H3')
    # par = {'example': ex}
    # config = CegisConfig(**par)
    # count = CounterExampleFinder(config)
    # zone = Zone(shape='ball', center=[0, 0], r=1)
    # count.get_extremum_scipy(zone, 'x1 + x2')
    # count.find_counterexample([False] * 8, [])
    # data = [[3, 1], [-1, 1], [1, 2], [1, 0]]
    # data = [np.array(e) for e in data]
    # count.get_ellipsoid(data)
    # count.get_center([0, 0], expr=sp.sympify('-x1-x2+1'), zone=zone)
