import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 50  # Grid size, can be adjusted
size = (N + 2, N + 2)
u = np.zeros(size)
v = np.zeros(size)
u_prev = np.zeros(size)
v_prev = np.zeros(size)
dens = np.zeros(size)
dens_prev = np.zeros(size)

def IX(i, j):
    return i + (N + 2) * j

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
    project(N, u, v, u0, v0)
    u0, u = u, u0
    v0, v = v, v0
    advect(N, 1, u, u0, u0, v0, dt)
    advect(N, 2, v, v0, u0, v0, dt)
    project(N, u, v, u0, v0)

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
    if b == 1:
        x[0, 1:N+1] = -x[1, 1:N+1]
        x[N+1, 1:N+1] = -x[N, 1:N+1]
    else:
        x[0, 1:N+1] = x[1, 1:N+1]
        x[N+1, 1:N+1] = x[N, 1:N+1]
    
    if b == 2:
        x[1:N+1, 0] = -x[1:N+1, 1]
        x[1:N+1, N+1] = -x[1:N+1, N]
    else:
        x[1:N+1, 0] = x[1:N+1, 1]
        x[1:N+1, N+1] = x[1:N+1, N]

    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, N+1] = 0.5 * (x[1, N+1] + x[0, N])
    x[N+1, 0] = 0.5 * (x[N, 0] + x[N+1, 1])
    x[N+1, N+1] = 0.5 * (x[N, N+1] + x[N+1, N])

def draw_dens(N, dens):
    plt.imshow(dens[1:N+1, 1:N+1], cmap='viridis', origin='lower', extent=[0, N, 0, N])
    #plt.colorbar()

def get_from_UI(dens_prev, u_prev, v_prev):
    # Clear the previous values
    dens_prev.fill(0)
    u_prev.fill(0)
    v_prev.fill(0)
    
    # Add a density source at the center
    center_x = N // 2
    center_y = N // 2
    dens_prev[center_x, center_y] = 100.0  # Example density value

    # Apply gravity to the y-velocity
    v_prev[1:N+1, 1:N+1] -= 0.1  # Example gravity value

# Simulation parameters
visc = 0.001
diff = 0.0001
dt = 0.1

fig, ax = plt.subplots()

def init():
    dens.fill(0)
    draw_dens(N, dens)

def update(frame):
    get_from_UI(dens_prev, u_prev, v_prev)
    vel_step(N, u, v, u_prev, v_prev, visc, dt)
    dens_step(N, dens, dens_prev, u, v, diff, dt)
    ax.clear()
    draw_dens(N, dens)
    return ax

ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=False)

plt.show()
