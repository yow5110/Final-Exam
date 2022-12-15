import matplotlib.pyplot as plt
import numpy as np
import ode  # this is the ode.py module we wrote in class, containing move_Euler() and move_RK2()

epsilon=16e-4
sigma=2.4
mass=1.
nparticles = 2 # Total of 2 particles
R=10 # Our simulation box is a square that spans -R to +R in both the x and y directions.
dt = 0.4


def flj(x,epsilon=4*4.e-4,sigma=6.):    
    """Compute Lennard-Jones forice as a function of inter-atomic distance"""
    # Task 1, calculate LJ force here.
    return ?

def distance(r1,r2):
    """Computes the distance (a scalar) and displacement (a vector) between two particles"""
    dr = r1-r2
    return np.sqrt(sum(dr**2)),dr

def calc_forces(positions, forc):
    """
    This function takes as input a 2D numpy array of dimensions nparticles x 2, 
    containing the x and y coordinates of n particles,

    and a function forc() that computes a force as a function of the distance
    between pairs of particles. 
    
    This functions returns a 2D numpy array of dimensions nparticles x 2,
    containing forces fx and fy acting on each particle.
    """
    forces = np.zeros([2,2])
    dist , dr =  distance(positions[1,:],positions[0,:])
    forces[0,:] = -forc(dist,epsilon,sigma) * dr / dist
    forces[1,:] = +forc(dist,epsilon,sigma) * dr / dist
    return  forces

def f(y):
    """ 
    This function takes as input
    y    =  [x1, vx1, z1, vz1, x2, vx2, z2, vz2]
    
    It returns the time derivative of every argument, in the same order
    ydot = [vx1, ax1, vz1, az1, vx2, ax2, vz2, az2] 
    """
    positions = y[::2].reshape(2,2)
    a = calc_forces( positions ,flj) / mass
    ydot = y.copy()
    
    ydot[1] = a[0,0] #ax1
    ydot[5] = a[1,0] #ax2
    ydot[3] = a[0,1] #az1
    ydot[7] = a[1,1] #az2

    #Task 1, complete populating the rest of ydot with velocities
    ydot[0] = y[?] #vx1
    ydot[4] = y[?] #vx2
    ydot[2] = y[?] #vz1
    ydot[6] = y[?] #vz2
    
    return ydot

# positions of 2 particles, one at -1,0, the other at 1,1
xz = np.array([-1,0,1,1]) 
# velocities of 2 particles, both static initially.
vxz = np.array([0,0,0,0])
# y is a 1D array including x1, vx1, z1, vz1, x2, vx2, z2, vz2.
y = np.vstack([xz,vxz]).T.flatten()

fig, ax = plt.subplots(1)
ax.set_aspect('equal', 'box')
ax.set_xlim(-R, R)
ax.set_ylim(-R, R)
a,=ax.plot([],[],'go')


import matplotlib.animation as ani
def animate(i):
    global y
    # Advance time one step forward
    y = ode.move_RK2(f, y, dt)
    
    #Task 1, set up reflective boundary conditions here:
    if y[?] < ?:
        y[?] = ?

    
    # This plots two circles at (x1,z1) and (x2,z2)
    # which translates to (y[0], y[4]) and (y[2], y[6]).
    # set_data() requires all x values as the first argument,
    # and all z values as the second argument.
    a.set_data([y[0], y[4]],[y[2], y[6]] ) 
    return a,

anim = ani.FuncAnimation(fig, animate, frames=1000, interval=1, blit=True, repeat=True)




