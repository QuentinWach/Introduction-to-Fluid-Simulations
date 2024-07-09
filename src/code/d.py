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
    dt0 = dt * N
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            if s[i, j] == 0:  # Only advect fluid cells
                x = i - dt0 * u[i, j]
                y = j - dt0 * v[i, j]
                x = np.clip(x, 0.5, N - 1.5)
                y = np.clip(y, 0.5, N - 1.5)
                i0, i1 = int(x), int(x) + 1
                j0, j1 = int(y), int(y) + 1
                s1, s0 = x - i0, i1 - x
                t1, t0 = y - j0, j1 - y
                
                # Check if any of the four surrounding cells is a solid
                if s[i0, j0] == 0 and s[i0, j1] == 0 and s[i1, j0] == 0 and s[i1, j1] == 0:
                    d[i, j] = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                               s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]))
                else:
                    # If any surrounding cell is solid, don't advect (keep current value)
                    d[i, j] = d0[i, j]
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
N = 150  # grid size
fps = 15 
dt = 1.5 / fps
steps = fps*3
diff = 0.0001  # diffusion rate
visc = 0.00001  # viscosity

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
s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = 1
# Add some solid objects inside the fluid
s[30:40, 30:40] = 1
s[100:120, 80:100] = 1

# Create the figure
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

# Create a custom colormap with red for obstacles
cmap = plt.get_cmap('bone_r').copy()
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

def rand_inkjets():
    x = np.random.randint(1, N-1)
    y = np.random.randint(1, N-1)
    if s[x, y] == 0:
        dens[x-4:x+6, y-4:y+6] += 0.9
        v[x-4:x+6, y-4:y+6] += 2 * np.random.randn()
        u[x-4:x+6, y-4:y+6] += 2 * np.random.randn()

def top_down_inkjets():
    x = N//2
    y = 5
    
    if s[x, y] == 0:
        dens[y:y+3, x-30:x+30] += 0.9
        u[y:y+3, x-30:x+30] += 1

def gravity(strength=0.01):
    u[2:N-2,2:N-2] += strength  # Changed from u to v for downward gravity

def global_disturbances(strength=1.0):
    u[:, :] += strength * (np.random.rand(N, N) * 0.1 - 0.05)
    v[:, :] += strength * (np.random.rand(N, N) * 0.1 - 0.05)

def global_dissipation():
    dens[:,:] *= 0.99

# Animation function
def update(frame):
    if frame % 2 == 0:
        # Randomly add drops with velocities
        #rand_inkjets()

        # Add inkjets from top to bottom
        top_down_inkjets()


    # Add gravity
    #gravity()	
    # Add random velocity disturbances
    #global_disturbances()
    # Dissipate densities
    global_dissipation()
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
anim = FuncAnimation(fig, update, frames=steps, interval=1000//fps, blit=False)
print("Rendering animation...")
writer = FFMpegWriter(fps=fps, bitrate=1800)
anim.save("fluid_simulation.mp4", writer=writer)
print("Finished rendering.")