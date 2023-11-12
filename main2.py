import matplotlib.pyplot as plt
import numpy as np


def rk4(f, x0, y0, h, n):
    """
    Implements the Runge-Kutta method of the 4th order for solving the differential equation y' = f(x, y).

    Parameters:
    f : function
        The function f(x, y) representing the differential equation y' = f(x, y).
    x0 : float
        The initial value of x.
    y0 : float
        The initial value of y.
    h : float
        The step size.
    n : int
        The number of steps to take.

    Returns:
    x : array
        An array containing the x values.
    y : array
        An array containing the y values.
    """

    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = x0
    y[0] = y0

    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x[i + 1] = x[i] + h

    return x, y


# Example usage:
def f(x, y):
    return x * np.sqrt(y)


x0 = 0
y0 = 1
h = 0.1
n = 100

x, y = rk4(f, x0, y0, h, n)

# Plot the solution
plt.plot(x, y, label='Numerical solution')


# Plot the analytical solution
def y_analytical(x):
    return (1 / 16) * (x ** 2 + 4) ** 2


x_analytical = np.linspace(x0, x0 + h * n, n + 1)
y_analytical = y_analytical(x_analytical)

plt.plot(x_analytical, y_analytical, label='Analytical solution')

# Plot the initial condition
plt.plot([x0], [y0], 'ro', label='Initial condition')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.show()
