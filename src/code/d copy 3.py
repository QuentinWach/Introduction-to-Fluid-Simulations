import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.colors as colors

def set_bnd(b, x, s):
    # Set boundary conditions for solid walls
    x[s == 1] = 0  # Set values inside solid objects to 0
    # Set boundary conditions for edges
    if b == 1:  # x-component of velocity
        x[0, :] = -x[1, :]
        x[-1, :] = -x[-2, :]
    elif b == 2:  # y-component of velocity
        x[:, 0] = -x[:, 1]
        x[:, -1] = -x[:, -2]
    else:  # scalar (density)
        x[0, :] = x[1, :]
        x[-1, :] = x[-2, :]
        x[:, 0] = x[:, 1]
        x[:, -1] = x[:, -2]
    # Set corners
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

def lin_solve(b, x, x0, a, c, s):
    for _ in range(20):
        x[1:-1, 1:-1] = np.where(s[1:-1, 1:-1] == 0,
                                 (x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])) / c,
                                 x[1:-1, 1:-1])
        set_bnd(b, x, s)

def diffuse(b, x, x0, diff, dt, s):
    a = dt * diff * N * N
    lin_solve(b, x, x0, a, 1 + 4 * a, s)

# Update the project function as suggested in the previous response
def project(u, v, p, div, s):
    h = 1.0 / N
    div[1:-1, 1:-1] = -0.5 * h * (u[2:, 1:-1] - u[:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, :-2])
    p.fill(0)
    set_bnd(0, div, s)
    set_bnd(0, p, s)
    lin_solve(0, p, div, 1, 4, s)
    
    u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
    v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
    
    # Zero out velocities pointing into solid cells
    u[1:-1, 1:-1] = np.where(s[1:-1, 1:-1] == 1, 0, u[1:-1, 1:-1])
    u[2:, 1:-1] = np.where(s[1:-1, 1:-1] == 1, 0, u[2:, 1:-1])
    v[1:-1, 1:-1] = np.where(s[1:-1, 1:-1] == 1, 0, v[1:-1, 1:-1])
    v[1:-1, 2:] = np.where(s[1:-1, 1:-1] == 1, 0, v[1:-1, 2:])
    
    set_bnd(1, u, s)
    set_bnd(2, v, s)

def advect(b, d, d0, u, v, dt, s):
    N = d.shape[0]
    dt0 = dt * N
    for i in range(1, N-1):
        for j in range(1, N-1):
            if s[i, j] == 1:
                continue
            
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            
            if x < 0.5: x = 0.5
            if x > N + 0.5: x = N + 0.5
            i0, i1 = int(x), int(x) + 1
            
            if y < 0.5: y = 0.5
            if y > N + 0.5: y = N + 0.5
            j0, j1 = int(y), int(y) + 1
            
            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1
            
            if i0 < 1: i0 = 1
            if i1 > N - 1: i1 = N - 1
            if j0 < 1: j0 = 1
            if j1 > N - 1: j1 = N - 1
            
            d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    set_bnd(b, d, s)

def step(u, v, u0, v0, dens, dens0, s):
    diffuse(1, u0, u, visc, dt, s)
    diffuse(2, v0, v, visc, dt, s)
    project(u0, v0, u, v, s)
    advect(1, u, u0, u0, v0, dt, s)
    advect(2, v, v0, u0, v0, dt, s)
    project(u, v, u0, v0, s)
    diffuse(0, dens0, dens, diff, dt, s)
    advect(0, dens, dens0, u, v, dt, s)



# Set parameters
np.random.seed(42)
N = 200  # Increased grid size for better resolution
fps = 24
dt = 1.0 / fps
steps = fps*3
diff = 0.0001
visc = 0.00001

# Initialize variables
u = np.zeros((N, N))
v = np.zeros((N, N))
u_prev = np.zeros((N, N))
v_prev = np.zeros((N, N))
dens = np.zeros((N, N))
dens_prev = np.zeros((N, N))

# Create s-matrix (0 for fluid, 1 for solid)
s = np.zeros((N, N))

# Add solid walls
#s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = 1

# Add a circular obstacle in the center
center = N // 2
radius = N // 8
y, x = np.ogrid[:N, :N]
mask = (x - center)**2 + (y - center)**2 <= radius**2
s[mask] = 1

# Create the figure
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

# Create a custom colormap with red for obstacles
cmap = plt.get_cmap('viridis').copy()
cmap.set_bad(color='red')

# Use the custom colormap for the imshow
im = ax.imshow(dens, cmap=cmap, vmin=0, vmax=1, interpolation='bicubic')

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
    global u, v, dens
    
    # Add fluid from the left side
    dens[80:120, 1:3] += 0.4
    v[80:120, 1:3] += 2.  # Horizontal velocity to the right
    
    # Add some random perturbations
    #u += np.random.randn(N, N) * 0.05
    #v += np.random.randn(N, N) * 0.05

    # move to the right
    #v[:, :] += 0.2
    
    # Dissipate densities
    dens *= 0.95
    
    # Step forward in time
    step(u, v, u_prev, v_prev, dens, dens_prev, s)
    
    # Create a masked array where obstacles are masked
    masked_dens = np.ma.array(dens, mask=s)
    
    # Draw the image
    im.set_array(masked_dens)
    
    # Update
    print(f"Frame {frame}/{steps}. {np.round((frame/steps)*100,1)}% done.")
    return [im]

# Run and save the animation as an MP4 file
anim = FuncAnimation(fig, update, frames=steps, blit=True)
print("Rendering animation...")
writer = FFMpegWriter(fps=fps, bitrate=1800)
anim.save("fluid_simulation_ball_obstacle.mp4", writer=writer)
print("Finished rendering.")