"""
Eulerian Fluid in 2D.
author: @QuentinWach
date: June 28, 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        self.u = np.zeros(self.numCells, dtype=np.float32)
        self.v = np.zeros(self.numCells, dtype=np.float32)
        self.newU = np.zeros(self.numCells, dtype=np.float32)
        self.newV = np.zeros(self.numCells, dtype=np.float32)
        self.p = np.zeros(self.numCells, dtype=np.float32)
        self.s = np.zeros(self.numCells, dtype=np.float32)
        self.m = np.ones(self.numCells, dtype=np.float32)
    
    def integrate(self, dt, gravity):
        """
        For every point i,j in the grid, we update the velocity field
        by adding the gravity force to the y-component of the velocity
        making sure to differentiate between fluid and solid cells.
        """
        n = self.numY
        for i in range(1, self.numX):
            for j in range(1, self.numY-1):
                if self.s[i*n + j] != 0.0 and self.s[i*n + j-1] != 0.0:
                    self.v[i*n + j] += gravity * dt
    
    def solveIncompressibility(self, numIters, dt, overRelaxation):
        """
        
        """
        n = self.numY
        cp = self.density * self.h / dt

        for _ in range(numIters):
            for i in range(1, self.numX-1):
                for j in range(1, self.numY-1):
                    if self.s[i*n + j] == 0.0:
                        continue

                    s = self.s[i*n + j]
                    sx0 = self.s[(i-1)*n + j]
                    sx1 = self.s[(i+1)*n + j]
                    sy0 = self.s[i*n + j-1]
                    sy1 = self.s[i*n + j+1]
                    s = sx0 + sx1 + sy0 + sy1
                    if s == 0.0:
                        continue

                    div = self.u[(i+1)*n + j] - self.u[i*n + j] + self.v[i*n + j+1] - self.v[i*n + j]

                    p = -div / s
                    p *= overRelaxation
                    self.p[i*n + j] += cp * p

                    self.u[i*n + j] -= sx0 * p
                    self.u[(i+1)*n + j] += sx1 * p
                    self.v[i*n + j] -= sy0 * p
                    self.v[i*n + j+1] += sy1 * p
    
    def extrapolate(self):
        """
        """
        n = self.numY
        for i in range(self.numX):
            self.u[i*n + 0] = self.u[i*n + 1]
            self.u[i*n + self.numY-1] = self.u[i*n + self.numY-2]
        for j in range(self.numY):
            self.v[0*n + j] = self.v[1*n + j]
            self.v[(self.numX-1)*n + j] = self.v[(self.numX-2)*n + j]
    
    def sampleField(self, x, y, field):
        """
        """
        n = self.numY
        h = self.h
        h1 = 1.0 / h
        h2 = 0.5 * h

        x = max(min(x, self.numX * h), h)
        y = max(min(y, self.numY * h), h)

        dx = 0.0
        dy = 0.0

        if field == U_FIELD:
            f = self.u
            dy = h2
        elif field == V_FIELD:
            f = self.v
            dx = h2
        elif field == S_FIELD:
            f = self.m
            dx = h2
            dy = h2

        x0 = min(int((x-dx)*h1), self.numX-1)
        tx = ((x-dx) - x0*h) * h1
        x1 = min(x0 + 1, self.numX-1)

        y0 = min(int((y-dy)*h1), self.numY-1)
        ty = ((y-dy) - y0*h) * h1
        y1 = min(y0 + 1, self.numY-1)

        sx = 1.0 - tx
        sy = 1.0 - ty

        val = sx*sy * f[x0*n + y0] + tx*sy * f[x1*n + y0] + tx*ty * f[x1*n + y1] + sx*ty * f[x0*n + y1]

        return val
    
    def advectVel(self, dt):
        """
        """
        self.newU = np.copy(self.u)
        self.newV = np.copy(self.v)

        n = self.numY
        h = self.h
        h2 = 0.5 * h

        for i in range(1, self.numX):
            for j in range(1, self.numY):
                if self.s[i*n + j] != 0.0 and self.s[(i-1)*n + j] != 0.0 and j < self.numY - 1:
                    x = i*h
                    y = j*h + h2
                    u = self.u[i*n + j]
                    v = (self.v[(i-1)*n + j] + self.v[i*n + j] + self.v[(i-1)*n + j+1] + self.v[i*n + j+1]) * 0.25
                    x = x - dt*u
                    y = y - dt*v
                    u = self.sampleField(x, y, U_FIELD)
                    self.newU[i*n + j] = u
                if self.s[i*n + j] != 0.0 and self.s[i*n + j-1] != 0.0 and i < self.numX - 1:
                    x = i*h + h2
                    y = j*h
                    u = (self.u[i*n + j-1] + self.u[i*n + j] + self.u[(i+1)*n + j-1] + self.u[(i+1)*n + j]) * 0.25
                    v = self.v[i*n + j]
                    x = x - dt*u
                    y = y - dt*v
                    v = self.sampleField(x, y, V_FIELD)
                    self.newV[i*n + j] = v

        self.u = np.copy(self.newU)
        self.v = np.copy(self.newV)
    
    def advectSmoke(self, dt):
        self.newM = np.copy(self.m)

        n = self.numY
        h = self.h
        h2 = 0.5 * h

        for i in range(1, self.numX-1):
            for j in range(1, self.numY-1):
                if self.s[i*n + j] != 0.0:
                    u = (self.u[i*n + j] + self.u[(i+1)*n + j]) * 0.5
                    v = (self.v[i*n + j] + self.v[i*n + j+1]) * 0.5
                    x = i*h + h2 - dt*u
                    y = j*h + h2 - dt*v
                    self.newM[i*n + j] = self.sampleField(x, y, S_FIELD)

        self.m = np.copy(self.newM)
    
    def simulate(self, dt, gravity, numIters, overRelaxation):
        self.integrate(dt, gravity)
        self.p.fill(0.0)
        self.solveIncompressibility(numIters, dt, overRelaxation)
        self.extrapolate()
        self.advectVel(dt)
        self.advectSmoke(dt)


def setup_scene(sceneNr=0):
    scene = {
        'gravity': -9.81,
        'dt': 1.0 / 120.0,
        'numIters': 100,
        'frameNr': 0,
        'overRelaxation': 1.9,
        'obstacleX': 0.0,
        'obstacleY': 0.0,
        'obstacleRadius': 0.15,
        'paused': False,
        'sceneNr': sceneNr,
        'showObstacle': False,
        'showStreamlines': False,
        'showVelocities': False,
        'showPressure': False,
        'showSmoke': True,
        'fluid': None
    }

    scene['overRelaxation'] = 1.9
    scene['dt'] = 1.0 / 60.0
    scene['numIters'] = 40

    res = 100
    if sceneNr == 0:
        res = 50
    elif sceneNr == 3:
        res = 200

    simHeight = 1.0
    simWidth = 1.0
    domainHeight = 1.0
    domainWidth = domainHeight / simHeight * simWidth
    h = domainHeight / res

    numX = int(domainWidth / h)
    numY = int(domainHeight / h)

    density = 1000.0
    fluid = Fluid(density, numX, numY, h)

    n = fluid.numY

    if sceneNr == 0:  # tank
        for i in range(fluid.numX):
            for j in range(fluid.numY):
                s = 1.0  # fluid
                if i == 0 or i == fluid.numX-1 or j == 0:
                    s = 0.0  # solid
                fluid.s[i*n + j] = s

        scene['gravity'] = -9.81
        scene['showPressure'] = True
        scene['showSmoke'] = False
        scene['showStreamlines'] = False
        scene['showVelocities'] = False

    elif sceneNr == 1 or sceneNr == 3:  # vortex shedding
        inVel = 2.0
        for i in range(fluid.numX):
            for j in range(fluid.numY):
                s = 1.0  # fluid
                if i == 0 or j == 0 or j == fluid.numY-1:
                    s = 0.0  # solid
                fluid.s[i*n + j] = s

                if i == 1:
                    fluid.u[i*n + j] = inVel

        pipeH = 0.1 * fluid.numY
        minJ = int(0.5 * fluid.numY - 0.5*pipeH)
        maxJ = int(0.5 * fluid.numY + 0.5*pipeH)

        for j in range(minJ, maxJ):
            fluid.m[j] = 0.0

        scene['gravity'] = 0.0
        scene['showPressure'] = False
        scene['showSmoke'] = True
        scene['showStreamlines'] = False
        scene['showVelocities'] = False

        if sceneNr == 3:
            scene['dt'] = 1.0 / 120.0
            scene['numIters'] = 100
            scene['showPressure'] = True

    elif sceneNr == 2:  # paint
        scene['gravity'] = 0.0
        scene['overRelaxation'] = 1.0
        scene['showPressure'] = False
        scene['showSmoke'] = True
        scene['showStreamlines'] = False
        scene['showVelocities'] = False
        scene['obstacleRadius'] = 0.1

    scene['fluid'] = fluid
    return scene


# Animation function
def update(frameNr, scene, im):
    if not scene['paused']:
        scene['fluid'].simulate(scene['dt'], scene['gravity'], scene['numIters'], scene['overRelaxation'])

    smoke_density = scene['fluid'].m.reshape((scene['fluid'].numX, scene['fluid'].numY))
    im.set_array(smoke_density.T)
    return [im]


# Main
scene = setup_scene(0)
fig, ax = plt.subplots()
smoke_density = scene['fluid'].m.reshape((scene['fluid'].numX, scene['fluid'].numY))
im = ax.imshow(smoke_density.T, origin='lower', cmap='gray', animated=True)

ani = animation.FuncAnimation(fig, update, fargs=(scene, im), frames=200, interval=50, blit=True)
plt.show()
