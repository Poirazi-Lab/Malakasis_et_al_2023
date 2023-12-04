import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_table("./results/weights_per_epoch_asdf.txt",header=None, error_bad_lines=False)
df=df.drop(df.columns[[len(df.values[0])-1]],axis=1)



n = np.arange(0,1,0.01)
alpha = 10
beta =  1
up= 1
right = 0.5
maxval=0.01
minval=0.001


(maxval+minval)/2


y = []
for v in n:
    b = -(maxval-minval) / (1 + np.exp(-alpha*(v -right)) / beta) + maxval
    y.append(b)



for i in range(0,len(df.values),130):	
	fig,ax=plt.subplots(figsize=(10, 10), dpi=600)
	plt.grid(True)
	ax.plot(n,y, c='r')
	ax.set_xlabel('Weight Value',fontweight='bold',fontsize=12)
	ax.set_ylabel("Learning Rate",fontweight='bold',fontsize=12)
	
	ax2=ax.twinx()
	x=df.values[i]
	ax2.hist(x,100,range=(0,1),ec="k")
	ax2.plot([0.5,0.5],[0,300],c='black',ls='dashed')
		
	ax2.set_ylabel("Counts",fontweight='bold',fontsize=12)
	ax2.set_yticks(range(0,400,100))
	plt.title('Weight Distribution and Learning Rate Curve',fontweight='bold',fontsize=15)
	plt.savefig("./temp/Weights" + str(i) +".eps", dpi=600)
	#plt.show()
plt.close('all')
