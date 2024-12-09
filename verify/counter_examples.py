import time

import numpy as np
import sympy as sp
import cvxpy as cp
import matplotlib.pyplot as plt


def accelerate_hyperplane(points, point1, constraints, lam_constraints):
    t = sp.Symbol('t', positive=True, real=True)

    for point2 in points:
        z = point1 + t * (point2 - point1)
        res = []
        for lam in lam_constraints:
            res.append(lam(*z))



def get_hyperplane(point1: np.ndarray, point2: np.ndarray, constraints, lam_constraints):
    t = sp.Symbol('t', positive=True, real=True)
    z = point1 + t * (point2 - point1)
    ans = []
    for i, lam in enumerate(lam_constraints):
        res = sp.solve(sp.expand(lam(*z)), t)

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
            print(B.value, d.value)
            return True, B.value, d.value
        else:
            return False, None, None
    except:
        return False, None, None


def get_maximum_volume_ellipsoid(center, constraints, tangent_nums=10, counter_nums=100):
    points = get_points(center, tangent_nums)

    x = sp.symbols([f'x{i + 1}' for i in range(len(center))])
    lam_constraints = [sp.lambdify(x, constraints[i]) for i in range(len(constraints))]
    hyperplane = [get_hyperplane(center, point, constraints, lam_constraints) for point in points]
    hyperplane = [item for item in hyperplane if item is not None]
    hyperplane_vector = [item[1] for item in hyperplane]

    state, B, d = get_ellipsoid(len(center), hyperplane_vector)

    if len(center) == 2 and state:
        hyperplane_expr = [item[0] for item in hyperplane]
        # plot(hyperplane_expr, B, d)
    if state:
        n = len(center)
        u = np.random.randn(counter_nums, n)
        u = np.array([e / np.sqrt(sum(e ** 2)) * np.random.random() ** (1 / n) for e in u]).T
        ellipsoid = B @ u + d
        return True, list(ellipsoid.T)
    else:
        return False, None


def parse_equation(eq_str):
    # 定义符号变量
    x1, x2 = sp.symbols('x1 x2')

    # 使用 sympy 解析方程字符串
    eq = sp.sympify(eq_str)

    # 求解方程 ax1 + bx2 + c = 0 中 x2 的表达式
    solution = sp.solve(eq, x2)
    return solution[0]


def quadratic_function(x):
    return (1 / 3) * x ** 2 - (2 / 3) * x - 1


def plot(hyperplane, B, d):
    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), 2, edgecolor='r', fill=False)
    ax.add_patch(circle)

    theta = np.linspace(0, 2 * np.pi, 100)
    u = np.array([np.cos(theta), np.sin(theta)])  # 单位圆上的点

    # 使用矩阵 B 对点进行线性变换
    transformed_points = B @ u

    # 加上平移向量 d
    transformed_points += d
    plt.plot(transformed_points[0, :], transformed_points[1, :], label='Transformed Ellipse', color='b')

    equations = [str(item) for item in hyperplane]

    # 创建x1的范围
    x1_vals = np.linspace(-5, 5, 50)

    # 绘制每个方程的直线
    for eq_str in equations:
        # 解析方程并获得 x2 的表达式
        x2_expr = parse_equation(eq_str)

        # 将 x1_vals 代入 x2 的表达式
        x2_vals = np.array([float(x2_expr.subs('x1', val)) for val in x1_vals])

        # 绘制直线
        # plt.plot(x1_vals, x2_vals, label=f'{eq_str} = 0')

    x = np.linspace(-4, 4, 400)  # 从-2到4生成400个点
    # 计算对应的y值
    y = quadratic_function(x)
    # 绘制图像
    plt.plot(x, y, color='black')
    # 添加标题和标签
    plt.xlabel('x1')
    plt.ylabel('x2')

    # 添加网格和坐标轴
    # plt.grid(True)
    # plt.axhline(0, color='black', linewidth=0.5)  # x轴
    # plt.axvline(0, color='black', linewidth=0.5)  # y轴
    plt.ylim(-5, 5)
    # 显示图例
    # plt.legend()
    ax.set_aspect(1)
    # 显示图形
    plt.show()


if __name__ == '__main__':
    # np.random.seed(2027)
    x = sp.symbols([f'x{i + 1}' for i in range(2)])
    expr0 = -1 / 3 * x[0] ** 2 + 2 / 3 * x[0] + x[1] + 1
    expr1 = 4 - x[0] ** 2 - x[1] ** 2
    # print(expr0.coeff(x[0] ** 2))
    f0 = sp.lambdify(x, expr0)
    f1 = sp.lambdify(x, expr1)
    # get_hyperplane(np.array([0, 0]), np.array([1, 1]), [expr0, expr1], [f0, f1])
    get_maximum_volume_ellipsoid((0.5, 0.5), [expr0, expr1])
