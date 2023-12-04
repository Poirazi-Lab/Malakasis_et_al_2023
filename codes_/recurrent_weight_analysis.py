import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_table("./results/recurrent_weights_per_epoch_asdf.txt",header=None, error_bad_lines=False)
#df=df.drop(df.columns[[len(df.values[0])-1]],axis=1)


for i in range(0,len(df.values),10):	
	x=df.values[i]
	plt.figure(figsize=(10, 10), dpi=80)
	plt.grid(True)
	plt.hist(x,100,range=(0,1),ec="k",color='darkorchid')
	plt.ylim(0,200)
	plt.xlabel('Weight Value',fontweight='bold',fontsize=12)
	plt.ylabel("Counts",fontweight='bold',fontsize=12)
	plt.title('Recurrent Weight Distribution ' + str(i) + ' iterations',fontweight='bold',fontsize=15)
	plt.savefig("./temp/RecurrentWeights" + str(i),bbox_inches='tight')
	plt.show()

