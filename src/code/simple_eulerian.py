"""
A minimal version of a 2D eulerian fluid simulation
with plenthy of comments based on the paper
"Real-Time Fluid Dynamics for Games" by Jos Stam.
author: @QuentinWach
date: July 1, 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# grid size
N = 50
# position vectors
x = np.zeros((N, N))
y = np.zeros((N, N))
# velocties
u = np.zeros((N, N))
v = np.zeros((N, N))
# densities
d = np.zeros((N, N))
# object
s = np.zeros((N, N))



class Fluid:
    def __init__():
        # sources of densities (fluid or solid objects)
        s = np.zeros((N, N))
        pass

    def sim():
        for t in range(100):
            # make a step
            # visualize the state             
            pass
        pass

    def step():
        # add forces

        # diffuse velocities

        # move

        pass

    def add_force():
        pass

    def diffuse():
        pass

    def move():
        pass



    

