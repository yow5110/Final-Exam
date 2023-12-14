import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
import myode  # this is the ode.py module we wrote in class, containing move_Euler() and move_RK2()

epsilon= 4e-4 * 27.211 * 1.6e-19 # in J
bohr = 0.53e-10 
sigma=2.4 * bohr # in m
mass= 40 * 1e-3/6.02e23 # in kg

gridsize = 4
nparticles = gridsize**2 
R = 10 * bohr # Our simulation box is a square that spans -R to +R in both the x and y directions.

k = 4*epsilon * 18/2**(1/3)/sigma**2
omega = np.sqrt(k/mass)

period = 2*np.pi/omega
dt = 0.01 * period


def flj(x):    
    """Compute Lennard-Jones force as a function of inter-atomic distance"""
    return -4*epsilon*(-12*sigma**12/x**13 \
                       +6*sigma**6/x**7)
        

def calc_forces(position):
    """
    This function takes as input a 2D numpy array of dimensions nparticles x 2, 
    containing the x and y coordinates of n particles,
    
    This function returns a 2D numpy array of dimensions nparticles x 2,
    containing forces fx and fy acting on each particle.
    """
    
    forces  = np.zeros((nparticles,2))
    
    for i in range(nparticles):
        for j in range(i+1, nparticles):

            x1 = position[i,0]
            z1 = position[i,1]
            x2 = position[j,0]
            z2 = position[j,1]
            r12 =  np.sqrt((x1-x2)**2 + (z1-z2)**2)
            f = flj(r12)

            forces[i,0] +=  f * (x1-x2)/r12 
            forces[i,1] +=  f * (z1-z2)/r12
            forces[j,0] -= forces[i,0]
            forces[j,1] -= forces[i,1]
            
    return forces
    
def diffeq(y):
    """ 
    This function takes as input
    y    =  [x1, z1, x2, z2, ...., vx1, vz1, vx2, vz2, ....]
    
    It returns the time derivative of every argument, in the same order
    ydot =  [vx1, vz1, vx2, vz2, ...., ax1, az1, ax2, az2, ....]
    """
    position = y[0:nparticles*2].reshape(nparticles,2).copy()
    a = calc_forces( position) / mass
    ydot = y.copy()
    ydot[:2*nparticles] = y[2*nparticles:] 
    ydot[2*nparticles:] = a.flatten()
    return ydot

#create xgrid and zgrid
xpos = np.linspace(-0.9*R, 0.9*R, gridsize)
zpos = np.linspace(-0.9*R,0.9*R, gridsize)
xgrid, zgrid = np.meshgrid(xpos, zpos)

# initial positions
y = np.zeros(4*nparticles)
y[0:2*nparticles:2] = xgrid.flatten()
y[1:2*nparticles:2] = zgrid.flatten()
 
# initial velocities
rng = np.random.default_rng(seed=1)
v = rng.uniform(-1,1, size=nparticles*2)*1e2
y[2*nparticles:] = v 


t_total = 500*dt
t_range = np.arange(0, t_total, dt)
y_list = []
    
for t in t_range:
    y = myode.move_RK2( diffeq, y, dt)

    for i in range(2*nparticles):
        if abs(y[i])>R:
            y[i+2*nparticles] = -y[i+2*nparticles]
             
    y_list.append(y)
    
y_list = np.array(y_list)    
    

fig, ax = plt.subplots(1)
ax.set_aspect('equal', 'box')
ax.set_xlim(-R,R)
ax.set_ylim(-R,R)
a, =  ax.plot([],[], 'go')

def animate(i):
    a.set_data( y_list[i, 0:2*nparticles:2]  ,\
               y_list[i, 1:2*nparticles:2] )
    return a,

anim = anim.FuncAnimation(fig, animate, \
                frames=len(t_range),\
                interval=0.1, repeat=True,\
                    blit=True)






    
    