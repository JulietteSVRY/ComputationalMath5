import numpy as np


def runge_kutta(f, x0, y0, h, x_end, n):

    n = int((x_end - x0) / h)

    x = np.arange(x0, x_end + h, h)
    y = np.zeros_like(x)

    y[0] = y0
    result = [(x0, y0)]
    for i in range(1, n+1):
        k1 = f(result[-1][0], result[-1][1])
        k2 = f(result[-1][0] + 0.5 * h, result[-1][1] + 0.5 * h * k1)
        k3 = f(result[-1][0] + 0.5 * h, result[-1][1] + 0.5 * h * k2)
        k4 = f(result[-1][0] + h, result[-1][1] + h * k3)
        y_next = result[-1][1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next = result[-1][0] + h
        result.append((x_next, y_next))

    return result
