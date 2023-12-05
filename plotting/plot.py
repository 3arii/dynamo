import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the differential equations
def system(y, t):
    dydt = -y[0]
    dxdt = 0.5 * y[0] - y[1]
    return [dydt, dxdt]

# Create a grid of values for the vector field plot
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
U, V = system([X, Y], 0)

# Plot the vector field
plt.quiver(X, Y, U, V, scale=20, color='blue', width=0.007)

# Define initial conditions and plot trajectories
initial_conditions = [[-1, 1], [0, 2], [1, -1]]

for ic in initial_conditions:
    # Integrate the system of ODEs
    t_span = np.linspace(0, 10, 100)
    sol = odeint(system, ic, t_span)

    # Plot the trajectory
    plt.plot(sol[:, 1], sol[:, 0], label=f'Initial condition: {ic}')

# Set plot labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Vector Field and Trajectories')
plt.grid(True)
plt.show()
