import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns



data_file = "./results/test_activity_asdf.txt"

df=pd.read_table(data_file)

digit1='1'
digit2='2'

active_thresh=20


dfd1=df[df['LABEL']==digit1]
c=0
overlap_list_d1=[]
overlap_list_d2=[]

activetab_d1=np.zeros((80,int(dfd1.shape[0]/800)))





for i in range(800,dfd1.shape[0]+1,800):
	temp_dfd1=dfd1.iloc()[c:i]
	for j in range(0,i-c,10):		
		active=int(int(temp_dfd1[temp_dfd1["BRANCHID"]==str(j)]["NEURON_SPIKES"])>=active_thresh)
		activetab_d1[int(j/10),int(c/800)]=active
		
	c=i

overlap_d1=np.nan_to_num(np.sum(activetab_d1[40:],axis=0)/np.sum(activetab_d1,axis=0))



dfd2=df[df['LABEL']==digit2]
c=0
activetab_d2=np.zeros((80,int(dfd2.shape[0]/800)))


for i in range(800,dfd2.shape[0]+1,800):
	temp_dfd2=dfd2.iloc()[c:i]
	for j in range(0,i-c,10):		
		active=int(int(temp_dfd2[temp_dfd2["BRANCHID"]==str(j)]["NEURON_SPIKES"])>=active_thresh)
		activetab_d2[int(j/10),int(c/800)]=active
		
	c=i

overlap_d2=np.nan_to_num(np.sum(activetab_d2[:40],axis=0)/np.sum(activetab_d2,axis=0))


overlap=np.hstack((overlap_d1,overlap_d2))


print("Total mean overlap:", np.mean(overlap))

plt.figure(figsize=(10,12))
plt.hist(overlap,bins=(np.arange(0,1.2,0.1)-0.05),color='blue',edgecolor='black')
plt.xlabel('Overlap',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.ylim(0,410)
plt.title('Active neuron overlap per test image histogram',fontweight='bold',fontsize=15)
plt.savefig("./temp/Active_overlap_total",bbox_inches='tight')
plt.show()



print("Mean overlap for first class:", np.mean(overlap_d1))

plt.figure(figsize=(10,12))
plt.hist(overlap_d1,bins=(np.arange(0,1.2,0.1)-0.05),color='blue',edgecolor='black')
plt.xlabel('Overlap',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.ylim(0,410)
plt.title('Active neuron overlap per test image for class 1 histogram',fontweight='bold',fontsize=15)
plt.savefig("./temp/Active_overlap_class_1",bbox_inches='tight')
plt.show()




print("Mean overlap for second class:", np.mean(overlap_d2))

plt.figure(figsize=(10,12))
plt.hist(overlap_d2,bins=(np.arange(0,1.2,0.1)-0.05),color='blue',edgecolor='black')
plt.xlabel('Overlap',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.ylim(0,410)
plt.title('Active neuron overlap per test image for class 2 histogram',fontweight='bold',fontsize=15)
plt.savefig("./temp/Active_overlap_class_2",bbox_inches='tight')
plt.show()

