import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

U_FIELD = 0
V_FIELD = 1
S_FIELD = 2

class Fluid:
    def __init__(self, density, numX, numY, h):
        self.density = density
        self.numX = numX + 2
        self.numY = numY + 2
        self.numCells = self.numX * self.numY
        self.h = h
        self.u = np.zeros((self.numX, self.numY), dtype=np.float32)
        self.v = np.zeros((self.numX, self.numY), dtype=np.float32)
        self.newU = np.zeros((self.numX, self.numY), dtype=np.float32)
        self.newV = np.zeros((self.numX, self.numY), dtype=np.float32)
        self.p = np.zeros((self.numX, self.numY), dtype=np.float32)
        self.s = np.ones((self.numX, self.numY), dtype=np.float32)
        self.m = np.zeros((self.numX, self.numY), dtype=np.float32)
        self.newM = np.zeros((self.numX, self.numY), dtype=np.float32)
        
        # Add rectangular obstacle
        obstacle_width = self.numX // 10
        obstacle_height = self.numY // 4
        obstacle_x = self.numX // 2 - obstacle_width // 2
        obstacle_y = self.numY // 2 - obstacle_height // 2
        self.s[obstacle_x:obstacle_x+obstacle_width, obstacle_y:obstacle_y+obstacle_height] = 0.0
        
        # Add density source
        self.source_x = obstacle_x - 5
        self.source_y = self.numY // 2

    def add_density_source(self):
        self.m[self.source_x, self.source_y] = 1.0

    def integrate(self, dt, gravity):
        for i in range(1, self.numX):
            for j in range(1, self.numY - 1):
                if self.s[i, j] != 0.0 and self.s[i, j-1] != 0.0:
                    self.v[i, j] += gravity * dt

    def solve_incompressibility(self, num_iters, dt):
        cp = self.density * self.h / dt

        for _ in range(num_iters):
            for i in range(1, self.numX - 1):
                for j in range(1, self.numY - 1):
                    if self.s[i, j] == 0.0:
                        continue

                    sx0 = self.s[i-1, j]
                    sx1 = self.s[i+1, j]
                    sy0 = self.s[i, j-1]
                    sy1 = self.s[i, j+1]
                    s = sx0 + sx1 + sy0 + sy1
                    if s == 0.0:
                        continue

                    div = self.u[i+1, j] - self.u[i, j] + self.v[i, j+1] - self.v[i, j]
                    p = -div / s
                    p *= self.over_relaxation  # Note: You need to define over_relaxation
                    self.p[i, j] += cp * p

                    self.u[i, j] -= sx0 * p
                    self.u[i+1, j] += sx1 * p
                    self.v[i, j] -= sy0 * p
                    self.v[i, j+1] += sy1 * p

    def extrapolate(self):
        self.u[:, 0] = self.u[:, 1]
        self.u[:, -1] = self.u[:, -2]
        self.v[0, :] = self.v[1, :]
        self.v[-1, :] = self.v[-2, :]

    def sample_field(self, x, y, field):
        h = self.h
        h1 = 1.0 / h
        h2 = 0.5 * h

        x = max(min(x, self.numX * h), h)
        y = max(min(y, self.numY * h), h)

        dx = dy = 0.0

        if field == U_FIELD:
            f = self.u
            dy = h2
        elif field == V_FIELD:
            f = self.v
            dx = h2
        elif field == S_FIELD:
            f = self.m
            dx = dy = h2

        x0 = min(int((x - dx) * h1), self.numX - 1)
        tx = ((x - dx) - x0 * h) * h1
        x1 = min(x0 + 1, self.numX - 1)

        y0 = min(int((y - dy) * h1), self.numY - 1)
        ty = ((y - dy) - y0 * h) * h1
        y1 = min(y0 + 1, self.numY - 1)

        sx = 1.0 - tx
        sy = 1.0 - ty

        val = (sx * sy * f[x0, y0] +
               tx * sy * f[x1, y0] +
               tx * ty * f[x1, y1] +
               sx * ty * f[x0, y1])

        return val

    def avg_u(self, i, j):
        return (self.u[i, j-1] + self.u[i, j] +
                self.u[i+1, j-1] + self.u[i+1, j]) * 0.25

    def avg_v(self, i, j):
        return (self.v[i-1, j] + self.v[i, j] +
                self.v[i-1, j+1] + self.v[i, j+1]) * 0.25

    def advect_vel(self, dt):
        self.newU = self.u.copy()
        self.newV = self.v.copy()

        h = self.h
        h2 = 0.5 * h

        for i in range(1, self.numX):
            for j in range(1, self.numY):
                # u component
                if self.s[i, j] != 0.0 and self.s[i-1, j] != 0.0 and j < self.numY - 1:
                    x = i * h
                    y = j * h + h2
                    u = self.u[i, j]
                    v = self.avg_v(i, j)
                    x = x - dt * u
                    y = y - dt * v
                    u = self.sample_field(x, y, U_FIELD)
                    self.newU[i, j] = u

                # v component
                if self.s[i, j] != 0.0 and self.s[i, j-1] != 0.0 and i < self.numX - 1:
                    x = i * h + h2
                    y = j * h
                    u = self.avg_u(i, j)
                    v = self.v[i, j]
                    x = x - dt * u
                    y = y - dt * v
                    v = self.sample_field(x, y, V_FIELD)
                    self.newV[i, j] = v

        self.u = self.newU.copy()
        self.v = self.newV.copy()

    def advect_smoke(self, dt):
        self.newM = self.m.copy()

        h = self.h
        h2 = 0.5 * h

        for i in range(1, self.numX - 1):
            for j in range(1, self.numY - 1):
                if self.s[i, j] != 0.0:
                    u = (self.u[i, j] + self.u[i+1, j]) * 0.5
                    v = (self.v[i, j] + self.v[i, j+1]) * 0.5
                    x = i * h + h2 - dt * u
                    y = j * h + h2 - dt * v
                    self.newM[i, j] = self.sample_field(x, y, S_FIELD)

        self.m = self.newM.copy()

    def simulate(self, dt, gravity, num_iters):
        self.add_density_source()
        self.integrate(dt, gravity)
        self.p.fill(0.0)
        self.solve_incompressibility(num_iters, dt)
        self.extrapolate()
        self.advect_vel(dt)
        self.advect_smoke(dt)

# Animation setup
fig, ax = plt.subplots()

fluid = Fluid(1.0, 100, 100, 0.1)
fluid.over_relaxation = 1.9

im = ax.imshow(fluid.m, cmap='bone_r', animated=True, vmin=0, vmax=1)

def update(frame):
    fluid.simulate(0.1, 9.81, 20)
    im.set_array(fluid.m)
    return [im]

anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()