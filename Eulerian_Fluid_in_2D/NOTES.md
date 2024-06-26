# Eulerian Fluid Simulator

We will look at a 2D simulation here first. Though moving to 3D is quite trivial. It is _Eulerian_ because we use a grid rather than points for the computations.

We assume that:
1. Water is an [incompressable fluid]().
2. It has no viscosity (even though adding it would be rather easy).

Our velocity vector
$$
\vec{v} = 
\begin{bmatrix}
v_x \\
v_y \\
\end{bmatrix}
$$

are saved not within the centers of the cells (_"collocated"_ grid) but rather at the boundaries creating a so called _"staggered"_ grid.

The indices for the grid positions are notated as $i, j$.

### Velocity Update
Now, for all $i,j$ we update the velocity
$$
v_x^{i,j} \leftarrow v_x^{i,j} + \Delta t \cdot g
$$
with the gravity $g: -9,81\;$m/s for time-steps $\Delta t$ of e.g. $\frac{1}{30}\;$s.

>**Question**: This is the simplest form of integration called [Euler integration](). If you have ever worked with chaotic systems, you'll may know that this can lead to large errors quickly! So why does this work here? Or does it?

### Divergence (Total Outflow)
We calculate the total outflow of a cell as
$$
d \leftarrow v_x^{i,j+1}-v_x^{i,j} + v_y^{i+1,j} - v_y^{i,j}.
$$
If $d$ is positive, we have too much outflow. If it is negative, we have too much inflow. Only if $d = 0$ is our fluid as incompressible as we desire!

Thus we must force incompressibility!

### Forcing Incompressibility
First, we compute the divergence.
We can then handle obstacles or walls by fixing those velocity vectors. So for static object, that point of the border would be zero. But if it is moving this will of course impact the velocity and we can simulate how the fluid is being pushed around!

### General Case
We define the scalar value $s^{i,j}$ for each cell, where objects are zero and fluids 1. We update it as
$$
s \leftarrow  s^{i+1. j} + s^{i-1, j} + s^{i,j+1} + s^{i,j-1}
$$
and
$$\begin{align*}
v_x^{i,j} &\leftarrow &&v_x^{i,j} &&+ d \cdot s^{i-1,j}/s \\
v_x^{i+1,j} &\leftarrow &&v_x^{i+1,j} &&+ d \cdot s^{i+1,j}/s \\
v_y^{i,j} &\leftarrow &&v_y^{i,j} &&+ d \cdot s^{i,j+1}/s \\
v_y^{i,j+1} &\leftarrow &&v_y^{i,j+1} &&+ d \cdot s^{i,j+1}/s.
\end{align*}$$

What is $s$?

### Solving the Grid
Naturally, we want to solve the whole grid. One, and possibly the simplest method here is to use the [Gauss-Seidel method]():

For $n$ iterations and for all $i,j$, we compute the general case.

An issue here is that we access boundary cells outside of the grid! To resolve this problem, we can add border cells and set $s^{i,j} = 0$ for them to make them walls. Alternatively, we could copy the neighbor cells that are inside the grid.

### Measuring Pressure
We can also store a physical pressure value $p^{i,j}$ inside each cell!

For the $n$ iterations and all $i,j$, we can then additionally calculate it as
$$
    p^{i,j} \leftarrow p^{i,j} + \frac{d}{s}\cdot \frac{\rho \; h}{\Delta t},
$$
where $\rho$ is the density of the fluid and $h$ is the grid spacing.

While not necessary for the simulation, it provides us with some interesting information without much additional effort!

### Overrelaxation
While the Guass-Seidel method is very simple to implement, it needs more iterations than global methods. Here comes _"overrelaxation"_ into play.

We multuply the divergence by a scalar $1 \leq o \leq 2$
$$
d \leftarrow o\cdot(v_x^{i+1, j} - v_x^{i,j} + v_y^{i,j+1} - v_y^{i,j})
$$
e.g. $o=1.9$. Doing so increases the convergence of the method dramatically! It is very possible that the simulation will collapse and lead to an physically implausible result if we do not overrelax.

And the pressure values still remain correct!

### Semi-Lagrangian Advection
In the real world, fluids are made of particles. We don't have static grids like we assume here. It is merely a useful abstraction. But we still need to move the velocity values in the grid just like the velocity state is carried by the particles in the real world!

We resolve this with [semi-Lagrangian advection](). (Lagrangian rather than Eulerian because we consider particles rather than a grid.)

...

### Streamlines


