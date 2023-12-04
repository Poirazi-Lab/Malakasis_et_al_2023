import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


with open('./results/input_data_asdf.txt') as f:
	plt.figure(figsize=(10, 10), dpi=80) 
	c=0
	lines=f.readlines()
	for i in range(len(lines)):	
		c+=1
		spiketimes=[int(j) for j in lines[i].split(' ')[:-1]]
		plt.scatter(spiketimes,c*np.ones(len(spiketimes)),c='k',s=1)
		#plt.xlim(0,4000)
		plt.ylim(0,785)
		plt.yticks(range(0,780,40))
		plt.title("Raster plot of Input during Testing",fontweight='bold')
		plt.xlabel( "Time(ms)", fontweight='bold')
		plt.ylabel( "Neuron Index", fontweight='bold')
		if c%784==0:
			c=0
		
	plt.show()

		
		