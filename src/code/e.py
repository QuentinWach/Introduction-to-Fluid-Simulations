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
    
    # Handle boundaries
    x[1:-1, 1:-1, 0] = -x[1:-1, 1:-1, 1] if b == 3 else x[1:-1, 1:-1, 1]
    x[1:-1, 1:-1, -1] = -x[1:-1, 1:-1, -2] if b == 3 else x[1:-1, 1:-1, -2]
    
    x[1:-1, 0, 1:-1] = -x[1:-1, 1, 1:-1] if b == 2 else x[1:-1, 1, 1:-1]
    x[1:-1, -1, 1:-1] = -x[1:-1, -2, 1:-1] if b == 2 else x[1:-1, -2, 1:-1]
    
    x[0, 1:-1, 1:-1] = -x[1, 1:-1, 1:-1] if b == 1 else x[1, 1:-1, 1:-1]
    x[-1, 1:-1, 1:-1] = -x[-2, 1:-1, 1:-1] if b == 1 else x[-2, 1:-1, 1:-1]
    
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
    cRecip = 1.0 / c
    for _ in range(iter):
        x[1:-1, 1:-1, 1:-1] = (x0[1:-1, 1:-1, 1:-1] + a * (
            x[2:, 1:-1, 1:-1] + x[:-2, 1:-1, 1:-1] +
            x[1:-1, 2:, 1:-1] + x[1:-1, :-2, 1:-1] +
            x[1:-1, 1:-1, 2:] + x[1:-1, 1:-1, :-2]
        )) * cRecip
        set_bnd(b, x)

def diffuse(b, x, x0, diff, dt, iter):
    a = dt * diff * (x.shape[0] - 2) ** 2
    lin_solve(b, x, x0, a, 1 + 6 * a, iter)

def advect(b, d, d0, velocX, velocY, velocZ, dt):
    N = d.shape[0]
    Nfloat = N - 2
    dtx = dt * Nfloat
    dty = dt * Nfloat
    dtz = dt * Nfloat
    
    jfloat, ifloat, kfloat = np.meshgrid(np.arange(1, N-1), np.arange(1, N-1), np.arange(1, N-1))
    tmp1 = dtx * velocX[1:-1, 1:-1, 1:-1]
    tmp2 = dty * velocY[1:-1, 1:-1, 1:-1]
    tmp3 = dtz * velocZ[1:-1, 1:-1, 1:-1]
    x = ifloat - tmp1
    y = jfloat - tmp2
    z = kfloat - tmp3
    
    x = np.clip(x, 0.5, Nfloat + 0.5)
    y = np.clip(y, 0.5, Nfloat + 0.5)
    z = np.clip(z, 0.5, Nfloat + 0.5)
    i0 = x.astype(int)
    i1 = i0 + 1
    j0 = y.astype(int)
    j1 = j0 + 1
    k0 = z.astype(int)
    k1 = k0 + 1
    
    s1 = x - i0
    s0 = 1 - s1
    t1 = y - j0
    t0 = 1 - t1
    u1 = z - k0
    u0 = 1 - u1
    
    d[1:-1, 1:-1, 1:-1] = (
        s0 * (t0 * (u0 * d0[i0, j0, k0] + u1 * d0[i0, j0, k1]) +
              t1 * (u0 * d0[i0, j1, k0] + u1 * d0[i0, j1, k1])) +
        s1 * (t0 * (u0 * d0[i1, j0, k0] + u1 * d0[i1, j0, k1]) +
              t1 * (u0 * d0[i1, j1, k0] + u1 * d0[i1, j1, k1]))
    )
    set_bnd(b, d)

def project(velocX, velocY, velocZ, p, div):
    N = velocX.shape[0]
    
    div[1:-1, 1:-1, 1:-1] = -0.5 * (
        velocX[2:, 1:-1, 1:-1] - velocX[:-2, 1:-1, 1:-1] +
        velocY[1:-1, 2:, 1:-1] - velocY[1:-1, :-2, 1:-1] +
        velocZ[1:-1, 1:-1, 2:] - velocZ[1:-1, 1:-1, :-2]
    ) / N
    p[1:-1, 1:-1, 1:-1] = 0
    set_bnd(0, div)
    set_bnd(0, p)
    lin_solve(0, p, div, 1, 6, 4)
    
    velocX[1:-1, 1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) * N
    velocY[1:-1, 1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) * N
    velocZ[1:-1, 1:-1, 1:-1] -= 0.5 * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) * N
    set_bnd(1, velocX)
    set_bnd(2, velocY)
    set_bnd(3, velocZ)

def fluid_cube_step(cube):
    N = cube.size
    visc = cube.visc
    diff = cube.diff
    dt = cube.dt
    Vx = cube.Vx
    Vy = cube.Vy
    Vz = cube.Vz
    Vx0 = cube.Vx0
    Vy0 = cube.Vy0
    Vz0 = cube.Vz0
    s = cube.s
    density = cube.density
    
    diffuse(1, Vx0, Vx, visc, dt, 4)
    diffuse(2, Vy0, Vy, visc, dt, 4)
    diffuse(3, Vz0, Vz, visc, dt, 4)
    
    project(Vx0, Vy0, Vz0, Vx, Vy)
    
    advect(1, Vx, Vx0, Vx0, Vy0, Vz0, dt)
    advect(2, Vy, Vy0, Vx0, Vy0, Vz0, dt)
    advect(3, Vz, Vz0, Vx0, Vy0, Vz0, dt)
    
    project(Vx, Vy, Vz, Vx0, Vy0)
    
    diffuse(0, s, density, diff, dt, 4)
    advect(0, density, s, Vx, Vy, Vz, dt)

def add_density(cube, x, y, z, amount):
    cube.density[x, y, z] += amount

def add_velocity(cube, x, y, z, amountX, amountY, amountZ):
    cube.Vx[x, y, z] += amountX
    cube.Vy[x, y, z] += amountY
    cube.Vz[x, y, z] += amountZ


def create_animation(cube, frames=200):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        add_density(cube, cube.size//2, cube.size//2, cube.size//2, 200)
        add_velocity(cube, cube.size//2, cube.size//2, cube.size//2, 0, 0.1, 0)
        fluid_cube_step(cube)
        
        x, y, z = np.where(cube.density > 1)
        c = cube.density[x, y, z]
        scatter = ax.scatter(x, y, z, c=c, cmap='hot', alpha=0.5)
        ax.set_xlim(0, cube.size)
        ax.set_ylim(0, cube.size)
        ax.set_zlim(0, cube.size)
        ax.set_title(f'Frame {frame}')
        return scatter,
    
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    return anim

# Create and run the simulation
size = 50
cube = FluidCube(size, 0.0001, 0.0001, 0.1)
anim = create_animation(cube)
plt.show()