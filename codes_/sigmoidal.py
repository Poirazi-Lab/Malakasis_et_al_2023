import numpy as np
import matplotlib.pyplot as plt



x = np.arange(0,1,0.01)
alpha = 30
beta =  1
up= 1
right = 0.5
maxval=0.01
minval=0.001




y = []
for v in x:
    b = -(maxval-minval) / (1 + np.exp(-alpha*(v -right)) / beta) + maxval
    y.append(b)

plt.plot(x,y,c='r')
plt.plot([0,1],[(maxval+minval)/2,(maxval+minval)/2])
#plt.plot([0.5,0.5],[minval,maxval])
#plt.yticks(np.arange(minval,maxval + 0.001,0.001))
plt.grid(True)
plt.xlabel("Weight")
plt.ylabel("Learning Rate")
plt.title("Synaptic Learning Rate",fontweight="bold")
plt.show()


#y=-(0.01/(1.1 + np.exp(- 1000 * (x - 0.4)) / 1) - 1)


'''
LIMITATIONS:
1. MINVAL CANNOT BE GREATER THAN 0.01.
2. MODEL WORKS FOR MAXVAL VALUES <0.01.(FAILS FOR SMALLER)
3. 
	
'''

