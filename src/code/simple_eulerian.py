"""
A minimal version of a 2D eulerian fluid simulation
with plenthy of comments.
author: @QuentinWach
date: July 1, 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Fluid:
    def __init__(self, height=50, width=50, h=1):
        # space discretization
        x = np.zeros((height, width))
        print(x)
        # velocity field
        #v = np.zeros((height, width))
        # divergence
        #d =
        # type (solid or fluid?)
        #s = 
        # pressure
        #p =
        
    def integrate(self, gravity=True):
        """
        We assume normal gravity force acting on the fluid
        and integrate simply using Euler's method.
        """
        pass

Fluid()