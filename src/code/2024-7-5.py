"""
A minimal version of a 2D eulerian fluid simulation
with plenthy of comments based on the paper
"Real-Time Fluid Dynamics for Games" by Jos Stam.
author: @QuentinWach
date: 
July 1, 2024 -> Smoke simulation with density spawn
July 5, 2024 -> Ink drop simulation with correct non-compressibility
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation, FFMpegWriter

def set_bnd(b, x):
    # Set boundary conditions
    x[0, :] = -x[1, :] if b == 1 else x[1, :]
    x[-1, :] = -x[-2, :] if b == 1 else x[-2, :]
    x[:, 0] = -x[:, 1] if b == 2 else x[:, 1]
    x[:, -1] = -x[:, -2] if b == 2 else x[:, -2]
    # Set corners
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

def lin_solve(b, x, x0, a, c):
    for _ in range(20):
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])) / c
        set_bnd(b, x)

def diffuse(b, x, x0, diff, dt):
    a = dt * diff * N * N
    lin_solve(b, x, x0, a, 1 + 4 * a)

def project(u, v, p, div):
    div[1:-1, 1:-1] = -0.5 * (u[2:, 1:-1] - u[:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, :-2]) / N
    p.fill(0)
    set_bnd(0, div)
    set_bnd(0, p)
    lin_solve(0, p, div, 1, 4)
    
    u[1:-1, 1:-1] -= 0.5 * N * (p[2:, 1:-1] - p[:-2, 1:-1])
    v[1:-1, 1:-1] -= 0.5 * N * (p[1:-1, 2:] - p[1:-1, :-2])
    set_bnd(1, u)
    set_bnd(2, v)

def advect(b, d, d0, u, v, dt):
    dt0 = dt * N
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            x = np.clip(x, 0.5, N - 1.5)
            y = np.clip(y, 0.5, N - 1.5)
            i0, i1 = int(x), int(x) + 1
            j0, j1 = int(y), int(y) + 1
            s1, s0 = x - i0, i1 - x
            t1, t0 = y - j0, j1 - y
            d[i, j] = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                       s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]))
    set_bnd(b, d)

def step(u, v, u0, v0, dens, dens0):
    diffuse(1, u0, u, visc, dt)
    diffuse(2, v0, v, visc, dt)
    project(u0, v0, u, v)
    advect(1, u, u0, u0, v0, dt)
    advect(2, v, v0, u0, v0, dt)
    project(u, v, u0, v0)
    diffuse(0, dens0, dens, diff, dt)
    advect(0, dens, dens0, u, v, dt)

def add_density(x, y):
    dens[x-5:x+5, y-5:y+5] += 0.5

def add_velocity(x, y, vx, vy):
    u[x-5:x+5, y-5:y+5] += vx
    v[x-5:x+5, y-5:y+5] += vy

# Set parameters
np.random.seed(42)
N = 150  # grid size
fps = 24
dt = 2.0 / fps
steps = fps * 1
diff = 0.0001  # diffusion rate
visc = 0.00001  # viscosity

# Initialize variables
u = np.zeros((N, N))
v = np.zeros((N, N))
u_prev = np.zeros((N, N))
v_prev = np.zeros((N, N))
dens = np.zeros((N, N))
dens_prev = np.zeros((N, N))

# Create the figure
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
im = ax.imshow(dens, cmap='bone_r', vmin=0, vmax=1, interpolation='bicubic')
# Show no boundaries
ax.set_xticks([])
ax.set_yticks([])
# Show no box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.tight_layout()

# Animation function
def update(frame):
    # Randomly add drops with velocities
    if frame % 1 == 0:
        x = np.random.randint(1, N-1)
        y = np.random.randint(1, N-1)
        dens[x-4:x+6, y-4:y+6] += 0.9
        v[x-4:x+6, y-4:y+6] += 2 * np.random.randn()  # Add upward velocity
        u[x-4:x+6, y-4:y+6] += 2 * np.random.randn()  # Add upward velocity
    # Add gravity
    u[1:N-1,1:N-1] += 0.002
    # Add random velocity distrubrances
    u[:, :] += np.random.rand(N, N) * 0.1 - 0.05
    # Dissipate densities
    dens[:,:] *= 0.95
    # Step forward in time
    step(u, v, u_prev, v_prev, dens, dens_prev)
    # Draw the image
    im.set_array(dens)
    # Update
    print(f"Frame {frame}/{steps}. {np.round((frame/steps)*100,1)}% done.")
    return [im]

# Run and save the animation as an MP4 file
anim = FuncAnimation(fig, update, frames=steps, blit=False)
print("Rendering animation...")
writer = FFMpegWriter(fps=fps, bitrate=1800)
anim.save("fluid_simulation.mp4", writer=writer)
print("Finished rendering.")