import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import csv

"""
By Ian Wilhite 
Made for MEEN 363, Spring 2025
"""

# Constants
m = 70  # kg, jumper mass
g = 9.81  # m/s², gravity
l = 18  # m, bungee cord natural length
Da = 8  # Ns/m, air drag coefficient

# Initial conditions
z0 = 100 # m, Starting at the jump point
v0 = 0  # m/s, Initial velocity

# Time span
t_span = (0, 20)
t_eval = np.linspace(0, 20, 1000)

# Range of stiffness values
K_values = np.linspace(40, 240, 7)  # N/m
eta_values = np.linspace(0.15, 0.30, 1)  # Loss factor range
K_values = np.array([round(k, 1) for k in K_values])
eta_values = np.array([round(eta, 3) for eta in eta_values])

# Omega values
# Elementwise product of K_values and eta_values
omega_array = np.array([[(K * eta)**0.5 for eta in eta_values] for K in K_values])
# print(omega_array) 

# Equilibrium position
# # g * m = kx; x = m*g/k; x = 70*9.81/40; x = 17.145
equilibrium_position_array = z0 - l - m * g / K_values
# for i, K in enumerate(K_values):
#     plt.axhline(y=equilibrium_position_array[i], linestyle='--', label=f"Equilibrium K={K}")

def equations(t, y, K, eta):
    """ODE system for bungee jumping."""
    z, v = y  # Position and velocity
    if z > z0 - l:  # Free fall phase 
        dzdt = v
        dvdt = - g + (Da / m) * v
    else:  # Elastic phase
        omega_n = np.sqrt(K / m) 
        Dc = (K * eta) / omega_n
        D = Da + Dc
        dzdt = v
        dvdt = (- g * m - D * v + K * ((z0 - l) - z)) / m 
    
    return [dzdt, dvdt]

# Solve for different values of K and eta
# plt.figure(figsize=(10, 6))

max_acc = np.zeros((len(K_values), len(eta_values))) 
max_acc_times = np.zeros((len(K_values), len(eta_values)))
max_vel = np.zeros((len(K_values), len(eta_values)))
max_vel_times = np.zeros((len(K_values), len(eta_values)))
min_pos = np.zeros((len(K_values), len(eta_values)))
min_pos_times = np.zeros((len(K_values), len(eta_values)))


# Create subplots for position, velocity, and acceleration
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for i, K in enumerate(K_values):
    for j, eta in enumerate(eta_values):
        sol = solve_ivp(equations, t_span, [z0, v0], args=(K, eta), t_eval=t_eval)
        acceleration = np.gradient(sol.y[1], sol.t)
        max_acc_time = sol.t[np.argmax(np.abs(acceleration))]

        max_acc[i, j] = np.max(np.abs(acceleration))
        max_acc_times[i, j] = max_acc_time
        
        max_vel_time = sol.t[np.argmax(np.abs(sol.y[1]))]
        max_vel[i, j] = np.max(np.abs(sol.y[1]))
        max_vel_times[i, j] = max_vel_time
        
        min_pos[i, j] = np.min(sol.y[0])
        min_pos_time = sol.t[np.argmin(sol.y[0])]
        min_pos_times[i, j] = min_pos_time
        # plt.plot(sol.t, sol.y[0], label=f"K={K}, η={eta}")

        # Position plot
        axs[0].plot(sol.t, sol.y[0], label=f"K={K}, η={eta}")
        axs[0].set_ylabel("Position (m)")
        axs[0].legend()
        axs[0].grid()

        # # Velocity plot
        axs[1].plot(sol.t, sol.y[1], label=f"K={K}, η={eta}")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].legend()
        axs[1].grid()

        # # Acceleration plot
        axs[2].plot(sol.t, acceleration, label=f"K={K}, η={eta}")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Acceleration (m/s²)")
        axs[2].legend()
        axs[2].grid()

        # plt.tight_layout()
        # plt.show()

print(f'eqs array: {equilibrium_position_array}')
print(f'eta: {eta_values}')
print(f'K_vals: {K_values}')
print(f'max_acc: {max_acc}')
print(f'max_acc_times: {max_acc_times}')

# Free fall motion
t_free_fall = np.linspace(0, 20, 1000)
free_fall = z0 - 0.5 * g * t_free_fall**2 
free_fall[free_fall < 0] = 0
axs[0].plot(t_eval, free_fall, label="Free Fall", linestyle='--')

# bungee position
axs[0].axhline(y=z0 - l, color='k', linestyle=':', label="Bungee Position")

# plt.xlabel("Time (s)")
# plt.ylabel("Position (m)")
# plt.title("Bungee Jumper Motion")
plt.legend()
# plt.grid()
plt.tight_layout()
plt.show()


# Export data to CSV
with open('bungee_jump_data.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['eta', 'K', 'max_acc', 'max_acc_time', 'max_vel', 'max_vel_time', 'min_pos', 'min_pos_time'])
    for i, K in enumerate(K_values):
        for j, eta in enumerate(eta_values):
            csvwriter.writerow([eta, K, max_acc[i, j], max_acc_times[i, j], max_vel[i, j], max_vel_times[i, j], min_pos[i, j], min_pos_times[i, j]])