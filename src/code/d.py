"""
A minimal version of a 2D eulerian fluid simulation
with plenthy of comments based on the paper
"Real-Time Fluid Dynamics for Games" by Jos Stam.
author: @QuentinWach
date: July 1, 2024

TODO:
+ [ ] Make boundaries work
+ [ ] Make sure compressibility is enforced!
+ [ ] Refactor the code to make it more readable
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def create_boundary(N):
    # init all as a fluid
    s = np.ones((N + 2, N + 2))
    # top
    for i in range(0, 2):
        for j in range(0, N+2):
            s[i, j] = 0
    # bottom
    for i in range(N, N+2):
        for j in range(0, N+2):
            s[i, j] = 0
    # left
    for i in range(0, N+2):
        for j in range(0, 2):
            s[i, j] = 0
    # right
    for i in range(0, N+2):
        for j in range(N, N+2):
            s[i, j] = 0
    return s

def add_source(N, x, s, dt):
    x += dt * s

def diffuse(N, b, x, x0, diff, dt):
    a = dt * diff * N * N
    for _ in range(20):
        x[1:N+1, 1:N+1] = (x0[1:N+1, 1:N+1] + a * (x[0:N, 1:N+1] + x[2:N+2, 1:N+1] + x[1:N+1, 0:N] + x[1:N+1, 2:N+2])) / (1 + 4 * a)
        set_bnd(N, b, x)

def advect(N, b, d, d0, u, v, dt):
    dt0 = dt * N
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            if x < 0.5: x = 0.5
            if x > N + 0.5: x = N + 0.5
            i0 = int(x)
            i1 = i0 + 1
            if y < 0.5: y = 0.5
            if y > N + 0.5: y = N + 0.5
            j0 = int(y)
            j1 = j0 + 1
            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1
            d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    set_bnd(N, b, d)

def dens_step(N, x, x0, u, v, diff, dt):
    add_source(N, x, x0, dt)
    x0, x = x, x0
    diffuse(N, 0, x, x0, diff, dt)
    x0, x = x, x0
    advect(N, 0, x, x0, u, v, dt)

def vel_step(N, u, v, u0, v0, visc, dt):
    add_source(N, u, u0, dt)
    add_source(N, v, v0, dt)
    u0, u = u, u0
    diffuse(N, 1, u, u0, visc, dt)
    v0, v = v, v0
    diffuse(N, 2, v, v0, visc, dt)
    project(N, u, v, u0, v0)  # Pass the boundary array 's' to the project function
    u0, u = u, u0
    v0, v = v, v0
    advect(N, 1, u, u0, u0, v0, dt)
    advect(N, 2, v, v0, u0, v0, dt)
    project(N, u, v, u0, v0)  # Pass the boundary array 's' to the project function


def project(N, u, v, p, div):
    h = 1.0 / N
    div[1:N+1, 1:N+1] = -0.5 * h * (u[2:N+2, 1:N+1] - u[0:N, 1:N+1] + v[1:N+1, 2:N+2] - v[1:N+1, 0:N])
    p.fill(0)
    set_bnd(N, 0, div)
    set_bnd(N, 0, p)
    for _ in range(20):
        p[1:N+1, 1:N+1] = (div[1:N+1, 1:N+1] + p[0:N, 1:N+1] + p[2:N+2, 1:N+1] + p[1:N+1, 0:N] + p[1:N+1, 2:N+2]) / 4
        set_bnd(N, 0, p)
    u[1:N+1, 1:N+1] -= 0.5 * (p[2:N+2, 1:N+1] - p[0:N, 1:N+1]) / h
    v[1:N+1, 1:N+1] -= 0.5 * (p[1:N+1, 2:N+2] - p[1:N+1, 0:N]) / h
    set_bnd(N, 1, u)
    set_bnd(N, 2, v)

def set_bnd(N, b, x):
    for i in range(1, N + 1):
        x[0, i] = s[0, i] * (x[1, i] if b != 1 else -x[1, i])
        x[N+1, i] = s[N+1, i] * (x[N, i] if b != 1 else -x[N, i])
        x[i, 0] = s[i, 0] * (x[i, 1] if b != 2 else -x[i, 1])
        x[i, N+1] = s[i, N+1] * (x[i, N] if b != 2 else -x[i, N])
    
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, N+1] = 0.5 * (x[1, N+1] + x[0, N])
    x[N+1, 0] = 0.5 * (x[N, 0] + x[N+1, 1])
    x[N+1, N+1] = 0.5 * (x[N, N+1] + x[N+1, N])

def draw_dens(N, dens, ax, im):
    im.set_array(dens[1:N+1, 1:N+1])
    return im

def get_from_UI(dens_prev, u_prev, v_prev):
    # Clear the previous values
    dens_prev.fill(0)
    u_prev.fill(0)
    v_prev.fill(0)
    
    # Add a density source
    center_x = 25 
    center_y = N // 2
    radius = 10
    for i in range(center_x - radius, center_x + radius + 1):
        for j in range(center_y - radius, center_y + radius + 1):
            dens_prev[i, j] = 80.0  # Example density value

    # Apply gravity to the y-velocity
    u_prev[1:N+1, 1:N+1] += 0.1  # Example gravity value

    # Apply random forces to the x-velocity and y-velocity
    u_prev[1:N+1, 1:N+1] += 1. * (np.random.rand(N, N) * 2 - 1)
    v_prev[1:N+1, 1:N+1] += 3. * (np.random.rand(N, N) * 2 - 1)


# Simulation parameters
# Fluid properties
visc = 0.0005
diff = 0.0002
# Time step
dt = 0.1
fps = 24
steps = fps*4
# Grid size
N = 250
size = (N + 2, N + 2)
# Velocities at t
u = np.zeros(size)
v = np.zeros(size)
# Velcities at t-1
u_prev = np.zeros(size)
v_prev = np.zeros(size)
# Densities at t
dens = np.zeros(size)
# Densities at t-1
dens_prev = np.zeros(size)
# Solid object boundary at the corners of the sim
s = create_boundary(N)
# Plot the object grid
#plt.imshow(s, cmap='bone')
#plt.show()

def update(frame):
    get_from_UI(dens_prev, u_prev, v_prev)
    vel_step(N, u, v, u_prev, v_prev, visc, dt)
    dens_step(N, dens, dens_prev, u, v, diff, dt)
    print("Frame " + str(frame) + "/" + str(steps))
    return draw_dens(N, dens, ax, im)

fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
im = ax.imshow(dens[1:N+1, 1:N+1], cmap='bone_r', origin='lower', extent=[0, N, 0, N], vmin=0, vmax=100)
#fig.colorbar(im, ax=ax, orientation='vertical', label='Density (a.u.)')
#ax.set_title('Eulerian Fluid Simulation')
ax.set_xlabel('X (a.u.)')
ax.set_ylabel('Y (a.u.)')
fig.tight_layout()

def init():
    dens.fill(0)
    return draw_dens(N, dens, ax, im)

ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False)

# Save the animation as an MP4 file
print("Rendering animation...")
writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
ani.save("fluid_simulation.mp4", writer=writer)

print("Showing animation...")
plt.show()
