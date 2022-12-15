import matplotlib.pyplot as plt
import numpy as np
import ode  # this is the ode.py module we wrote in class, containing move_Euler() and move_RK2()

epsilon=16e-4
sigma=2.4
mass=1.
gridsize = 4  #We will later initialize the positions of 16 particles on a 4x4 grid
nparticles = gridsize**2 # Total of 16 particles
R=10 # Our simulation box is a square that spans -R to +R in both the x and y directions.
dt = 0.4


def flj(x,epsilon, sigma):    
    """Compute Lennard-Jones forice as a function of inter-atomic distance"""
    # Task 2, same LJ forces here copied from Task 1.
    return ?

def distance(r1,r2):
    """Computes the distance (a scalar) and displacement (a vector) between two particles"""
    dr=r1-r2
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
    forces = np.zeros(positions.shape)
    for i in range(nparticles):
        for j in range(i+1,nparticles):
            dist,dr=distance(positions[j,:],positions[i,:])
            ftmp=forc(dist,epsilon,sigma)*dr/dist
            forces[i,:]-=ftmp[:]
            forces[j,:]+=ftmp[:]
    return  forces

def f(y):
    """ 
    This function takes as input a 1D array of length 4*nparticles
    y    =  [x1, vx1, z1, vz1, x2, vx2, z2, vz2, ........]
    
    It returns the time derivative of every argument, in the same order
    ydot = [vx1, ax1, vz1, az1, vx2, ax2, vz2, az2, ........].
    Also 4*nparticles in length.
    """
    #Task 2, complete the missing parts of f()
    positions = ?
    a = calc_forces(positions,flj)
    ydot = y.copy()
    ydot[?] = a[?]/mass # all x accelerations: ax1,ax2,ax3,...
    ydot[?] = a[?]/mass # all z accelerations: az1,az2,az3,...
    ydot[?] = y[?]      # all x velocities: vx1,vx2,vx3,...
    ydot[?] = y[?]      # all z velocities: vz1,vz2,vz3,...
    
    return ydot


# positions of all particles, initialized on a 4x4 grid
xline = np.linspace(-0.9*R, 0.9*R, gridsize)
zline = np.linspace(-0.9*R,0.9*R, gridsize)
xgrid, zgrid = np.meshgrid(xline, zline)
xz = np.vstack([xgrid.flatten(), zgrid.flatten() ]).T.flatten()


#Task 2, choose seed for initial velocities
rng = np.random.default_rng()
vxz = rng.uniform(-0.1,0.1, size=nparticles*2)
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
    
    # Task 2, finish unpacking y in preparation for 
    # setting up reflective boundary conditions next.
    
    x  = y[?] # 1D array of all x coordinates: x1,x2,x3...
    vx = y[?] # 1D array of all x velocities : vx1,vx2,vx3...
    z  = y[?] # 1D array of all z coordinates: z1,z2,z3...
    vz = y[?] # 1D array of all z velocities : vz1,vz2,vz3...
    
    # Task 2, set up reflective boundary conditions here.
    # Hint: You may use the list[ conditional ] format to quickly choose out-of-bounds coordinates
    # e.g. like a[a>0] we discussed in class.
    # Or you can use some if statements.
    
    



    # update plot. Yes, set_data can now be as clean as this!
    a.set_data(x ,z )
    return a,

anim = ani.FuncAnimation(fig, animate, frames=1000, interval=1, blit=True, repeat=True)

