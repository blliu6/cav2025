import time

import numpy as np
import sympy as sp
import cvxpy as cp
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool

from verify.plot_ellipsoid import plot


# import line_profiler
# from joblib import Parallel, delayed
#
# import os
#
# os.environ['LINE_PROFILE'] = '1'


# def expand(expr):
#     t = sp.Symbol('t', positive=True, real=True)
#     expr = sp.expand(expr)
#     ans = sp.roots(expr, t)
#     ans = [r for r, m in ans.items() if r.is_real and r > 0]
#     if len(ans) > 0:
#         return min(ans)
#     else:
#         return np.inf
#
#
# def compute_hyperplane(args):
#     t_value, pos, point1, point2, constraints = args
#     if t_value != np.inf:
#         tangent_point = point1 + t_value * (point2 - point1)
#
#         x = sp.symbols([f'x{i + 1}' for i in range(len(point1))])
#         diff = [sp.lambdify(x, sp.diff(constraints[pos], x[i]))(*tangent_point) for i in range(len(x))]
#         # print(diff)
#
#         cutting_plane = sp.expand(sum([diff[i] * (x[i] - tangent_point[i]) for i in range(len(x))]))
#         b = -sum([diff[i] * tangent_point[i] for i in range(len(x))])
#         a = np.array(diff)
#
#         lam_cutting_plane = sp.lambdify(x, cutting_plane)
#         if lam_cutting_plane(*point1) > 0:
#             return cutting_plane, (np.array([-a]), b)
#         else:
#             return -cutting_plane, (np.array([a]), -b)
#     else:
#         return None
#
#
# def accelerate_hyperplane(points, point1, constraints, lam_constraints):
#     t = sp.Symbol('t', positive=True, real=True)
#
#     res = []
#     for point2 in points:
#         z = point1 + t * (point2 - point1)
#         for lam in lam_constraints:
#             res.append(lam(*z))
#     pool = Pool(mp.cpu_count() // 3)
#     res = pool.map(expand, res)
#
#     # res = [expand(item) for item in res]
#
#     # res = Parallel(n_jobs=mp.cpu_count() // 3)(delayed(expand)(item) for item in res)
#     res = np.array(res).reshape(-1, len(constraints))
#     res_min = np.min(res, axis=1)
#     res_pos = np.argmin(res, axis=1)
#     res_pos = [int(item) for item in res_pos]
#
#     ans = [(res_min[i], res_pos[i], point1, points[i], constraints) for i in range(len(points))]
#     ans = pool.map(compute_hyperplane, ans)
#     return ans


def get_hyperplane(point1: np.ndarray, point2: np.ndarray, constraints, lam_constraints):
    t = sp.Symbol('t', positive=True, real=True)
    z = point1 + t * (point2 - point1)
    ans = []
    for i, lam in enumerate(lam_constraints):
        expr = sp.expand(lam(*z))
        res = sp.roots(expr, t)
        res = [r for r, m in res.items() if r.is_real and r > 0]

        if len(res) > 0:
            ans.append((min(res), i))

    if len(ans) > 0:
        t_value, pos = min(ans)
        tangent_point = point1 + t_value * (point2 - point1)

        x = sp.symbols([f'x{i + 1}' for i in range(len(point1))])
        diff = [sp.lambdify(x, sp.diff(constraints[pos], x[i]))(*tangent_point) for i in range(len(x))]
        # print(diff)

        cutting_plane = sp.expand(sum([diff[i] * (x[i] - tangent_point[i]) for i in range(len(x))]))
        b = -sum([diff[i] * tangent_point[i] for i in range(len(x))])
        a = np.array(diff)

        lam_cutting_plane = sp.lambdify(x, cutting_plane)
        if lam_cutting_plane(*point1) > 0:
            return cutting_plane, (np.array([-a]), b)
        else:
            return -cutting_plane, (np.array([a]), -b)
    else:
        return None


def get_points(center, nums):
    # s = np.random.randn(nums, len(center))
    s = [np.random.randn(len(center)) for i in range(nums)]
    s = np.array([e / np.sqrt(sum(e ** 2)) * np.sqrt(1) for e in s])
    s = s + center
    return s


def get_ellipsoid(n, hyperplane):
    B = cp.Variable((n, n), PSD=True)
    d = cp.Variable((n, 1))

    con = []
    for a, b in hyperplane:
        con.append(cp.norm(B @ a.T) + a @ d <= b)

    obj = cp.Minimize(-cp.log_det(B))
    prob = cp.Problem(obj, con)
    try:
        prob.solve(solver=cp.MOSEK)
        if prob.status == 'optimal':
            # print(B.value, d.value)
            return True, B.value, d.value
        else:
            return False, None, None
    except:
        return False, None, None


def get_maximum_volume_ellipsoid(center, constraints, tangent_nums=20, counter_nums=100, args=None):
    points = get_points(center, tangent_nums)

    x = sp.symbols([f'x{i + 1}' for i in range(len(center))])
    lam_constraints = [sp.lambdify(x, constraints[i]) for i in range(len(constraints))]
    # hyperplane = accelerate_hyperplane(points, center, constraints, lam_constraints)

    hyperplane = [get_hyperplane(center, point, constraints, lam_constraints) for point in points]
    hyperplane = [item for item in hyperplane if item is not None]
    hyperplane_vector = [item[1] for item in hyperplane]

    state, B, d = get_ellipsoid(len(center), hyperplane_vector)

    hyperplane_expr = [item[0] for item in hyperplane]
    if len(center) == 2 and state and (args is not None):
        ex, bc = args
        plot(ex, bc, hyperplane_expr, B, d, center)
    if state:
        n = len(center)
        u = np.random.randn(counter_nums, n)
        u = np.array([e / np.sqrt(sum(e ** 2)) * np.random.random() ** (1 / n) for e in u]).T
        ellipsoid = B @ u + d
        return True, list(ellipsoid.T)
    else:
        return False, None
