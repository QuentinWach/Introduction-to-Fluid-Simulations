import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FluidCube:
    def __init__(self, size, diffusion, viscosity, dt):
        self.size = size
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity
        
        self.s = np.zeros((size, size, size))
        self.density = np.zeros((size, size, size))
        
        self.Vx = np.zeros((size, size, size))
        self.Vy = np.zeros((size, size, size))
        self.Vz = np.zeros((size, size, size))
        
        self.Vx0 = np.zeros((size, size, size))
        self.Vy0 = np.zeros((size, size, size))
        self.Vz0 = np.zeros((size, size, size))

def set_bnd(b, x):
    N = x.shape[0]
    
    # Handle edges
    x[1:-1, 0, :] = -x[1:-1, 1, :] if b == 2 else x[1:-1, 1, :]
    x[1:-1, -1, :] = -x[1:-1, -2, :] if b == 2 else x[1:-1, -2, :]
    
    x[0, 1:-1, :] = -x[1, 1:-1, :] if b == 1 else x[1, 1:-1, :]
    x[-1, 1:-1, :] = -x[-2, 1:-1, :] if b == 1 else x[-2, 1:-1, :]
    
    x[:, :, 0] = -x[:, :, 1] if b == 3 else x[:, :, 1]
    x[:, :, -1] = -x[:, :, -2] if b == 3 else x[:, :, -2]
    
    # Handle corners
    x[0, 0, 0] = 1/3 * (x[1, 0, 0] + x[0, 1, 0] + x[0, 0, 1])
    x[0, N-1, 0] = 1/3 * (x[1, N-1, 0] + x[0, N-2, 0] + x[0, N-1, 1])
    x[0, 0, N-1] = 1/3 * (x[1, 0, N-1] + x[0, 1, N-1] + x[0, 0, N-2])
    x[0, N-1, N-1] = 1/3 * (x[1, N-1, N-1] + x[0, N-2, N-1] + x[0, N-1, N-2])
    x[N-1, 0, 0] = 1/3 * (x[N-2, 0, 0] + x[N-1, 1, 0] + x[N-1, 0, 1])
    x[N-1, N-1, 0] = 1/3 * (x[N-2, N-1, 0] + x[N-1, N-2, 0] + x[N-1, N-1, 1])
    x[N-1, 0, N-1] = 1/3 * (x[N-2, 0, N-1] + x[N-1, 1, N-1] + x[N-1, 0, N-2])
    x[N-1, N-1, N-1] = 1/3 * (x[N-2, N-1, N-1] + x[N-1, N-2, N-1] + x[N-1, N-1, N-2])

def lin_solve(b, x, x0, a, c, iter):
    c_recip = 1.0 / c
    for _ in range(iter):
        x[1:-1, 1:-1, 1:-1] = (x0[1:-1, 1:-1, 1:-1] + a * (
            x[2:, 1:-1, 1:-1] + x[:-2, 1:-1, 1:-1] +
            x[1:-1, 2:, 1:-1] + x[1:-1, :-2, 1:-1] +
            x[1:-1, 1:-1, 2:] + x[1:-1, 1:-1, :-2]
        )) * c_recip
        set_bnd(b, x)

def diffuse(b, x, x0, diff, dt, iter):
    N = x.shape[0]
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x, x0, a, 1 + 6 * a, iter)

def project(velocX, velocY, velocZ, p, div, iter):
    N = velocX.shape[0]
    
    div[1:-1, 1:-1, 1:-1] = -0.5 * (
        velocX[2:, 1:-1, 1:-1] - velocX[:-2, 1:-1, 1:-1] +
        velocY[1:-1, 2:, 1:-1] - velocY[1:-1, :-2, 1:-1] +
        velocZ[1:-1, 1:-1, 2:] - velocZ[1:-1, 1:-1, :-2]
    ) / N
    p.fill(0)
    set_bnd(0, div)
    set_bnd(0, p)
    lin_solve(0, p, div, 1, 6, iter)
    
    velocX[1:-1, 1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) * N
    velocY[1:-1, 1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) * N
    velocZ[1:-1, 1:-1, 1:-1] -= 0.5 * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) * N
    
    set_bnd(1, velocX)
    set_bnd(2, velocY)
    set_bnd(3, velocZ)

def advect(b, d, d0, velocX, velocY, velocZ, dt):
    N = d.shape[0]
    
    dtx = dt * (N - 2)
    dty = dt * (N - 2)
    dtz = dt * (N - 2)
    
    x, y, z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
    
    x = x.astype(np.float64) - dtx * velocX
    y = y.astype(np.float64) - dty * velocY
    z = z.astype(np.float64) - dtz * velocZ
    
    x = np.clip(x, 0.5, N - 1.5)
    y = np.clip(y, 0.5, N - 1.5)
    z = np.clip(z, 0.5, N - 1.5)
    
    i0 = np.floor(x).astype(int)
    i1 = i0 + 1
    j0 = np.floor(y).astype(int)
    j1 = j0 + 1
    k0 = np.floor(z).astype(int)
    k1 = k0 + 1
    
    s1 = x - i0
    s0 = 1 - s1
    t1 = y - j0
    t0 = 1 - t1
    u1 = z - k0
    u0 = 1 - u1
    
    i0 = np.clip(i0, 0, N-1)
    i1 = np.clip(i1, 0, N-1)
    j0 = np.clip(j0, 0, N-1)
    j1 = np.clip(j1, 0, N-1)
    k0 = np.clip(k0, 0, N-1)
    k1 = np.clip(k1, 0, N-1)
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                d[i, j, k] = (
                    s0[i, j, k] * (
                        t0[i, j, k] * (u0[i, j, k] * d0[i0[i, j, k], j0[i, j, k], k0[i, j, k]] +
                                       u1[i, j, k] * d0[i0[i, j, k], j0[i, j, k], k1[i, j, k]]) +
                        t1[i, j, k] * (u0[i, j, k] * d0[i0[i, j, k], j1[i, j, k], k0[i, j, k]] +
                                       u1[i, j, k] * d0[i0[i, j, k], j1[i, j, k], k1[i, j, k]])
                    ) +
                    s1[i, j, k] * (
                        t0[i, j, k] * (u0[i, j, k] * d0[i1[i, j, k], j0[i, j, k], k0[i, j, k]] +
                                       u1[i, j, k] * d0[i1[i, j, k], j0[i, j, k], k1[i, j, k]]) +
                        t1[i, j, k] * (u0[i, j, k] * d0[i1[i, j, k], j1[i, j, k], k0[i, j, k]] +
                                       u1[i, j, k] * d0[i1[i, j, k], j1[i, j, k], k1[i, j, k]])
                    )
                )
    
    set_bnd(b, d)

def fluid_cube_step(cube):
    N = cube.size
    visc = cube.visc
    diff = cube.diff
    dt = cube.dt
    
    diffuse(1, cube.Vx0, cube.Vx, visc, dt, 4)
    diffuse(2, cube.Vy0, cube.Vy, visc, dt, 4)
    diffuse(3, cube.Vz0, cube.Vz, visc, dt, 4)
    
    project(cube.Vx0, cube.Vy0, cube.Vz0, cube.Vx, cube.Vy, 4)
    
    advect(1, cube.Vx, cube.Vx0, cube.Vx0, cube.Vy0, cube.Vz0, dt)
    advect(2, cube.Vy, cube.Vy0, cube.Vx0, cube.Vy0, cube.Vz0, dt)
    advect(3, cube.Vz, cube.Vz0, cube.Vx0, cube.Vy0, cube.Vz0, dt)
    
    project(cube.Vx, cube.Vy, cube.Vz, cube.Vx0, cube.Vy0, 4)
    
    diffuse(0, cube.s, cube.density, diff, dt, 4)
    advect(0, cube.density, cube.s, cube.Vx, cube.Vy, cube.Vz, dt)

def fluid_cube_add_density(cube, x, y, z, amount):
    cube.density[x, y, z] += amount

def fluid_cube_add_velocity(cube, x, y, z, amount_x, amount_y, amount_z):
    cube.Vx[x, y, z] += amount_x
    cube.Vy[x, y, z] += amount_y
    cube.Vz[x, y, z] += amount_z

# Simulation setup
N = 64
dt = 0.1
diff = 0.0001
visc = 0.00001

cube = FluidCube(N, diff, visc, dt)

# Visualization setup
fig, ax = plt.subplots()
im = ax.imshow(cube.density[:, :, N//2], cmap='hot', vmin=0, vmax=1)

def update(frame):
    # Add density and velocity
    fluid_cube_add_density(cube, N//2, N//2, N//2, 100)
    fluid_cube_add_velocity(cube, N//2, N//2, N//2, 0, 5, 0)  # Changed velocity direction
    
    fluid_cube_step(cube)
    
    # Update the plot
    slice_data = cube.density[:, :, N//2]
    im.set_array(slice_data)
    im.set_clim(vmin=np.min(slice_data), vmax=np.max(slice_data))  # Adjust color limits
    return [im]

# Create the animation
anim = FuncAnimation(fig, update, frames=40, interval=50, blit=True)
plt.show()