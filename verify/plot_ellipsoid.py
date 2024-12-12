import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle

from benchmarks.Examplers import Zone, Example


def plot(ex: Example, bc, hyperplane, B, d, center):
    if ex.continuous:
        draw_continuous(ex, bc)
    plt.plot(center[0], center[1], 'o', color='r')
    plot_line(hyperplane, ex.l1)
    plot_ellipsoid(B, d)
    plt.show()


def plot_ellipsoid(B, d):
    theta = np.linspace(0, 2 * np.pi, 100)
    u = np.array([np.cos(theta), np.sin(theta)])
    transformed_points = B @ u + d
    plt.plot(transformed_points[0, :], transformed_points[1, :], color='b')


def parse_equation(eq_str):
    # 定义符号变量
    x1, x2 = sp.symbols('x1 x2')

    # 使用 sympy 解析方程字符串
    eq = sp.sympify(eq_str)

    # 求解方程 ax1 + bx2 + c = 0 中 x2 的表达式
    solution = sp.solve(eq, x2)
    if len(solution) > 0:
        return solution[0]
    else:
        return None


def plot_line(hyperplane_expr, zone):
    if zone.shape == 'box':
        l, r = zone.low[0], zone.up[0]
    else:
        l, r = zone.center[0] - np.sqrt(zone.r), zone.center[0] + np.sqrt(zone.r)

    equations = [str(item) for item in hyperplane_expr]

    x1_vals = np.linspace(l, r, 50)

    for eq_str in equations:
        x2_expr = parse_equation(eq_str)
        if x2_expr is not None:
            x2_vals = np.array([float(x2_expr.subs('x1', val)) for val in x1_vals])

            plt.plot(x1_vals, x2_vals)


# def draw(self):
#     fig = plt.figure()
#     ax = plt.gca()
#
#     ax.add_patch(self.draw_zone(self.ex.l1, 'grey', 'local_1'))
#     ax.add_patch(self.draw_zone(self.ex.l2, 'blue', 'local_2'))
#     ax.add_patch(self.draw_zone(self.ex.I, 'g', 'init'))
#     ax.add_patch(self.draw_zone(self.ex.U, 'r', 'unsafe'))
#     ax.add_patch(self.draw_zone(self.ex.g1, 'bisque', 'guard_1'))
#     ax.add_patch(self.draw_zone(self.ex.g2, 'orange', 'guard_2'))
#
#     l1, l2 = self.ex.l1, self.ex.l2
#
#     self.plot_vector_field(l1, self.ex.f1, 'slategrey')
#     self.plot_vector_field(l2, self.ex.f2, 'cornflowerblue')
#
#     self.plot_barrier(l1, self.b1, 'orchid')
#     self.plot_barrier(l2, self.b2, 'purple')
#
#     plt.xlim(min(l1.low[0], l2.low[0]) - 1, max(l1.up[0], l2.up[0]) + 1)
#     plt.ylim(min(l1.low[1], l2.low[1]) - 1, max(l1.up[1], l2.up[1]) + 1)
#     ax.set_aspect(1)
#     plt.legend(loc='lower left')
#     plt.savefig(f'picture/{self.ex.name}_2d.png', dpi=1000, bbox_inches='tight')
#     plt.show()


def draw_continuous(ex, bc):
    fig = plt.figure()
    ax = plt.gca()

    ax.add_patch(draw_zone(ex.l1, 'black', 'local_1'))
    ax.add_patch(draw_zone(ex.I, 'g', 'init'))
    ax.add_patch(draw_zone(ex.U, 'r', 'unsafe'))

    l1 = ex.l1

    plot_vector_field(l1, ex.f1)

    plot_barrier(l1, bc, 'black')

    plt.xlim(l1.low[0] - 1, l1.up[0] + 1)
    plt.ylim(l1.low[1] - 1, l1.up[1] + 1)
    ax.set_aspect(1)
    # plt.legend()
    # plt.savefig(f'picture/{self.ex.name}_2d.png', dpi=1000, bbox_inches='tight')
    # plt.show()


def plot_barrier(zone, hx, color):
    low, up = zone.low, zone.up
    x = np.linspace(low[0], up[0], 100)
    y = np.linspace(low[1], up[1], 100)

    X, Y = np.meshgrid(x, y)

    s_x = sp.symbols(['x1', 'x2'])
    fun_hx = sp.lambdify(s_x, hx, 'numpy')
    value = fun_hx(X, Y)
    plt.contour(X, Y, value, 0, alpha=0.8, colors=color)


def plot_vector_field(zone: Zone, f, color='grey'):
    low, up = zone.low, zone.up
    xv = np.linspace(low[0], up[0], 10)
    yv = np.linspace(low[1], up[1], 10)
    Xd, Yd = np.meshgrid(xv, yv)

    DX, DY = f[0]([Xd, Yd]), f[1]([Xd, Yd])
    DX = DX / np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
    DY = DY / np.linalg.norm(DY, ord=2, axis=1, keepdims=True)

    plt.streamplot(Xd, Yd, DX, DY, linewidth=0.3,
                   density=0.8, arrowstyle='-|>', arrowsize=1, color=color)


def draw_zone(zone: Zone, color, label, fill=False):
    if zone.shape == 'ball':
        circle = Circle(zone.center, np.sqrt(zone.r), color=color, label=label, fill=fill, linewidth=1.5)
        return circle
    else:
        w = zone.up[0] - zone.low[0]
        h = zone.up[1] - zone.low[1]
        box = Rectangle(zone.low, w, h, color=color, label=label, fill=fill, linewidth=1.5)
        return box
