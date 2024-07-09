import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FluidCube:
    def __init__(self, size, diffusion, viscosity, dt):
        self.size = size
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity
        
        self.s = np.zeros((size, size))
        self.density = np.zeros((size, size))
        
        self.Vx = np.zeros((size, size))
        self.Vy = np.zeros((size, size))
        
        self.Vx0 = np.zeros((size, size))
        self.Vy0 = np.zeros((size, size))
        
        # Initialize solid mask (no obstacles by default)
        self.solid_mask = np.zeros((size, size), dtype=bool)

    def add_circular_obstacle(self, center_x, center_y, radius):
        y, x = np.ogrid[:self.size, :self.size]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        self.solid_mask[dist_from_center <= radius] = True

    def add_rectangular_obstacle(self, x1, y1, x2, y2):
        self.solid_mask[y1:y2+1, x1:x2+1] = True

def set_bnd(b, x, solid_mask):
    # Handle solid obstacles
    if b == 1:  # X-component
        x[solid_mask] = -x[np.roll(solid_mask, 1, axis=0)]
    elif b == 2:  # Y-component
        x[solid_mask] = -x[np.roll(solid_mask, 1, axis=1)]
    elif b == 0:  # Density
        x[solid_mask] = x[~solid_mask].mean()
    
    # Handle grid boundaries
    x[0] = x[1]
    x[-1] = x[-2]
    x[:, 0] = x[:, 1]
    x[:, -1] = x[:, -2]
    
    # Handle corners
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

def lin_solve(b, x, x0, a, c, iter, solid_mask):
    cRecip = 1.0 / c
    for _ in range(iter):
        x[1:-1, 1:-1] = np.where(solid_mask[1:-1, 1:-1], 0,
            (x0[1:-1, 1:-1] + a * (
                x[2:, 1:-1] + x[:-2, 1:-1] +
                x[1:-1, 2:] + x[1:-1, :-2]
            )) * cRecip
        )
        set_bnd(b, x, solid_mask)

def diffuse(b, x, x0, diff, dt, iter, solid_mask):
    a = dt * diff * (x.shape[0] - 2) ** 2
    lin_solve(b, x, x0, a, 1 + 4 * a, iter, solid_mask)

def advect(b, d, d0, velocX, velocY, dt, solid_mask):
    dtx = dt * (d.shape[0] - 2)
    dty = dt * (d.shape[0] - 2)
    
    jfloat, ifloat = np.meshgrid(np.arange(1, d.shape[0]-1), np.arange(1, d.shape[0]-1))
    tmp1 = dtx * velocX[1:-1, 1:-1]
    tmp2 = dty * velocY[1:-1, 1:-1]
    x = ifloat - tmp1
    y = jfloat - tmp2
    
    x = np.clip(x, 0.5, d.shape[0] - 1.5)
    y = np.clip(y, 0.5, d.shape[0] - 1.5)
    i0 = x.astype(int)
    i1 = i0 + 1
    j0 = y.astype(int)
    j1 = j0 + 1
    
    s1 = x - i0
    s0 = 1 - s1
    t1 = y - j0
    t0 = 1 - t1
    
    d[1:-1, 1:-1] = np.where(solid_mask[1:-1, 1:-1], 0,
        s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
        s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    )
    set_bnd(b, d, solid_mask)

def project(velocX, velocY, p, div, solid_mask):
    h = 1.0 / velocX.shape[0]
    div[1:-1, 1:-1] = -0.5 * h * (
        velocX[2:, 1:-1] - velocX[:-2, 1:-1] +
        velocY[1:-1, 2:] - velocY[1:-1, :-2]
    )
    p.fill(0)
    set_bnd(0, div, solid_mask)
    set_bnd(0, p, solid_mask)
    lin_solve(0, p, div, 1, 4, 4, solid_mask)
    
    velocX[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
    velocY[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
    
    set_bnd(1, velocX, solid_mask)
    set_bnd(2, velocY, solid_mask)

def fluid_cube_step(cube):
    visc = cube.visc
    diff = cube.diff
    dt = cube.dt
    Vx = cube.Vx
    Vy = cube.Vy
    Vx0 = cube.Vx0
    Vy0 = cube.Vy0
    s = cube.s
    density = cube.density
    solid_mask = cube.solid_mask
    
    diffuse(1, Vx0, Vx, visc, dt, 4, solid_mask)
    diffuse(2, Vy0, Vy, visc, dt, 4, solid_mask)
    
    project(Vx0, Vy0, Vx, Vy, solid_mask)
    
    advect(1, Vx, Vx0, Vx0, Vy0, dt, solid_mask)
    advect(2, Vy, Vy0, Vx0, Vy0, dt, solid_mask)
    
    project(Vx, Vy, Vx0, Vy0, solid_mask)
    
    diffuse(0, s, density, diff, dt, 4, solid_mask)
    advect(0, density, s, Vx, Vy, dt, solid_mask)
    
    # No need to explicitly set values inside obstacles to zero
    # as set_bnd will handle this correctly

def add_density(cube, size, x, y, amount):
    for i in range(x-size//2, x+size//2):
        for j in range(y-size//2, y+size//2):
            if not cube.solid_mask[j, i]:
                cube.density[j,i] += amount

def add_velocity(cube, size, x, y, amountX, amountY):
    if not cube.solid_mask[y, x]:
        cube.Vx[y-size//2:y+size//2, x-size//2:x+size//2] += amountX
        cube.Vy[y-size//2:y+size//2, x-size//2:x+size//2] += amountY

def create_animation(cube, frames=200, interval=50):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    # Show no boundaries
    ax.set_xticks([])
    ax.set_yticks([])
    # Show no box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.tight_layout()
        
    im = ax.imshow(cube.density, cmap='bone_r', animated=True, vmin=0, vmax=100, interpolation="bicubic", zorder=1)
    obstacle = ax.imshow(cube.solid_mask, cmap='magma', alpha=0.7, zorder=100)
    #fig.colorbar(im)

    # Add a cube object here at the center of the grid as an obstacle
    cube_size = cube.size // 3
    center = cube.size // 2
    cube.add_rectangular_obstacle(center - cube_size//2, center - cube_size//2, 
                                  center + cube_size//2, center + cube_size//2)
    
    def update(frame):
        add_density(cube, size=10, x=cube.size//8, y=cube.size//2, amount=50)
        add_velocity(cube, size=10, x=cube.size//8, y=cube.size//2, amountX=0, amountY=2)
        fluid_cube_step(cube)
        
        im.set_array(cube.density)
        obstacle.set_array(cube.solid_mask)  # Update the obstacle visualization
        return [im]
    
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    return anim

# Simulation parameters
size = 150
diffusion = 0.0001
viscosity = 0.0000001
fps = 15
dt = fps / 100
frames = fps * 1
interval = 1000 // fps

cube = FluidCube(size, diffusion, viscosity, dt)
anim = create_animation(cube, frames=frames, interval=interval)
plt.show()

