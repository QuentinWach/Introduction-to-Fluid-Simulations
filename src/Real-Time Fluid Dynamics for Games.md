# Real-Time Fluid Dynamics for Games


## The Physics of Fluids
$$
    \frac{\partial \vec{u}}{\partial t} = (\vec{u} \cdot \nabla) \vec{u} + v \nabla^2 \vec{u} + \vec{f} \\\\
    \frac{\partial \rho}{\partial t} = (\vec{u} \cdot \nabla) \rho + \kappa \nabla^2 \rho + S
$$


## A Fluid in a Box
|![Fig1](imgs/a1.png)|
|:-:| 
| **Figure 1.** *Computational grids considered in this paper. Both the density and the velocity are defined at the cell centers. The grid contains an extra layer of cells to account for the boundary conditions.* |


## Moving Densities
![Fig2](imgs/a2.png)
>Basic structure of the density solver. At every time step we resolve the three terms appearing on the right hand side of the density equation.

### Diffusion
![Fig3](imgs/a3.png)
>Through diffusion each cell exchanges density with its direct neighbors.

### Advection
|![Fig4](imgs/a4.png)|
|:-:| 
|**Figure 4.** *The advection step moves the density through a static velocity field.*|

|![Fig5](imgs/a5.png)|
|:-:|
|**Figure 5.** *The basic idea behind the advection step. Instead of moving the cell centers forward in time (b) through the velocity field shown in (a), we look for the particles which end up exactly at the cell centers by tracing backwards in time from the cell centers (c).*|

## Evolving Velocities
![Fig6](imgs/a6.png)