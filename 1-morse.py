import numpy as np
import matplotlib.pyplot as plt

epsilon= 4e-4 * 27.211 * 1.6e-19 # in J
bohr = 0.53e-10 
sigma=2.4 * bohr # in m


def flj(x):    
    return -4*epsilon*(-12*sigma**12/x**13 \
                       +6*sigma**6/x**7)
      
def ulj(x):    
    return 4*epsilon*(sigma**12/x**12 \
                       -sigma**6/x**6)
        
rrange = np.linspace(1,2,100)*sigma

fig,ax = plt.subplots(2, figsize=(6,6))
ax[0].plot(rrange, ulj(rrange), 'k--')
ax[1].plot(rrange, flj(rrange), 'k-' )

ax[0].axhline(-epsilon)
ax[1].axhline(0)

ax[0].set_ylabel('Potentials(r)')
ax[1].set_ylabel('Forces(r)')
ax[0].set_xlabel('r')
ax[1].set_xlabel('r')

fig.tight_layout()



