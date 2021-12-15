import matplotlib.pyplot as plt
import numpy as np
from sympy import *
from scipy.integrate import solve_ivp


def func(x, y):
    return 5 * y - 24 * y ** 3 - x ** 3 * y


def euler_method(a, b, y0, h):
    x = [a]
    y = [y0]
    xs, ys = symbols('xs ys')
    g = 1 + h * diff(func(xs, ys), ys)
    if abs(g.subs({xs: x[0], ys: y[0]})) <= 1:
        for i in range(1, int((b - a) / h) + 1):
            x.append(x[i - 1] + h)
            y.append(y[i - 1] + h * func(x[i - 1], y[i - 1]))
        return y
    else:
        return false  # "solution is unstable"


def implicit_euler_method(a, b, y0, h):
    x = [a]
    y = [y0]
    xs, ys = symbols('xs ys')
    g = 1 / (1 - h * diff(func(xs, ys), ys))
    # print("g = ", g.subs({xs: x[0], ys: y[0]}))
    if abs(g.subs({xs: x[0], ys: y[0]})) <= 1:
        for i in range(1, int((b - a) / h) + 1):
            x.append(x[i - 1] + h)
            xm = x[i - 1] + h
            ym = y[i - 1] + h * func(x[i - 1], y[i - 1])
            y.append(y[i - 1] + h * (func(x[i - 1], y[i - 1]) + func(xm, ym)) / 2)
        return y
    else:
        return false


def runge_kutta2_method(a, b, y0, h):
    x = [a]
    y = [y0]
    for i in range(1, int((b - a) / h) + 1):
        x.append(x[i - 1] + h)
        d1 = func(x[i - 1], y[i - 1])
        y.append(y[i - 1] + (h / 2) * (d1 + func(x[i - 1] + h, y[i - 1] + h * d1)))
    return y


def runge_kutta4_method(a, b, y0, h):
    x = [a]
    y = [y0]
    for i in range(1, int((b - a) / h) + 1):
        x.append(x[i - 1] + h)
        d1 = func(x[i - 1], y[i - 1])
        d2 = func(x[i - 1] + h / 2, y[i - 1] + d1 * h / 2)
        d3 = func(x[i - 1] + h / 2, y[i - 1] + d2 * h / 2)
        d4 = func(x[i - 1] + h, y[i - 1] + d3 * h)
        d_ = (d1 + 2 * d2 + 2 * d3 + d4) / 6
        y.append(y[i - 1] + h * d_)
    return y


def sol_with_scipy(a, b, h):
    x = np.arange(a, b + h, h)
    res = solve_ivp(func, [0, 1], [1], t_eval=x)
    return x, res.y[0]


def draw_graphs(a, b, res, h, lbl):
    xi = np.arange(a, b + h, h)
    plt.plot(xi, res, label=lbl)
    plt.plot(sol_with_scipy(a, b, h)[0], sol_with_scipy(a, b, h)[1], label="scipy")
    plt.scatter(xi, res, c='r')
    plt.scatter(xi, sol_with_scipy(a, b, h)[1], c='g')
    plt.legend()
    plt.grid()
    plt.show()
    plt.title('Error')
    plt.plot(xi, abs(res - sol_with_scipy(a, b, h)[1]))
    plt.grid()
    plt.show()


x0, xn = 0, 1
y0 = 1
hi = 0.01
print("euler: ", euler_method(x0, xn, y0, hi))
print("implicit euler: ", implicit_euler_method(x0, xn, y0, hi))
print("runge-kutta 2: ", runge_kutta2_method(x0, xn, y0, hi))
print("runge-kutta 4: ", runge_kutta4_method(x0, xn, y0, hi))
print("scipy: ", sol_with_scipy(x0, xn, hi)[1])

draw_graphs(x0, xn, euler_method(x0, xn, y0, hi), hi, "euler")
draw_graphs(x0, xn, implicit_euler_method(x0, xn, y0, hi), hi, "implicit euler")
draw_graphs(x0, xn, runge_kutta2_method(x0, xn, y0, hi), hi, "runge-kutta 2")
draw_graphs(x0, xn, runge_kutta4_method(x0, xn, y0, hi), hi, "runge-kutta 4")
