import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import interpolation_lab4
import math
from RungeKutta import runge_kutta
import colors as color

print('\n' + color.BOLD + color.RED + "Численное дифференцирование и задача Коши.\nМетод Рунге-Кутты 4-го порядка.", color.END)

print(color.YELLOW, "1. cos(x) - y", color.END)
print(color.YELLOW, "2. x^2 - y", color.END)
print(color.YELLOW, "3. x * y", color.END)
print(color.YELLOW, "4. x * y^2", color.END)

ans = int(input('Выберите номер функции для анализа: '))
answers = [1, 2, 3, 4]
while ans not in answers:
    ans = int(input('Неправильный ввод, попробуйте еще раз!'))


def runge_kutta(f, x0, y0, h, x_end):

    x = np.arange(x0, x_end + h, h)
    y = np.zeros_like(x)
    y[0] = y0

    for i in range(1, len(x)):
        k1 = h * f(x[i - 1], y[i - 1])
        k2 = h * f(x[i - 1] + h / 2, y[i - 1] + k1 / 2)
        k3 = h * f(x[i - 1] + h / 2, y[i - 1] + k2 / 2)
        k4 = h * f(x[i - 1] + h, y[i - 1] + k3)
        y[i] = y[i - 1] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x, y

# Define the differential equation
def f(x, y):
    return x**2 - y


# User input for initial conditions and range of independent variable
x0 = float(input("Введите начальное значение x: "))
y0 = float(input("Введите начальное значение y: "))
x_end = float(input("Введите конечное значение x: "))
h = float(input("Введите размер шага h: "))


# Analytical solution of the differential equation
def y_exact(x):
    return x**2 - (y0 - x0**2 + np.exp(-x))


x_rk, y_rk = runge_kutta(f, x0, y0, h, x_end)


x_exact = np.arange(x0, x_end + h, h / 10)
y_exact_spline = CubicSpline(x_exact, y_exact(x_exact))

y_rk_spline = CubicSpline(x_rk, y_rk)


fig, ax = plt.subplots()
ax.plot(x_exact, y_exact_spline(x_exact), label="Аналитическое решение")
plt.plot([x0], [y0], 'ro', label='Начальное значение')
ax.plot(x_rk, y_rk_spline(x_rk), label="Численное решение интерполяцией кубическими сплайнами")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Метод Рунге-Кутта 4-ого порядка")
ax.legend()
plt.show()
