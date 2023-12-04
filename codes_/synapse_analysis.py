import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns


# Input
data_file = "./results/synapse_analysis_asdf.txt"
#data_file = "/media/nikos/Elements/ClusterResults/results_250822_shapes_synapses_experiment/synapse_analysis_nikos_shapes_0_0_1_1_2000.txt"

# Delimiter
data_file_delimiter = '\t'

# The max column count a line in the file could have
largest_column_count = 0

# Loop the data lines
with open(data_file, 'r') as temp_f:
    # Read the lines
    lines = temp_f.readlines()

    for l in lines:
        # Count the column count for the current line
        column_count = len(l.split(data_file_delimiter)) + 1
        
        # Set the new most column count
        largest_column_count = column_count if largest_column_count < column_count else largest_column_count

# Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
column_names = [i for i in range(0, largest_column_count)]

# Read csv
df = pd.read_csv(data_file, header=None, delimiter=data_file_delimiter, names=column_names)
df=df.fillna(0)


info_table=df.values[:,:4]

weight_table=df.values[:,4:]

'''
neuronthres=np.ones((1,weight_table.shape[1]))*np.nan

c=0
for i in range(0,800,10):
	weight_table=np.insert(weight_table,i+c,neuronthres,0)
	weight_table=np.insert(weight_table,i+c+1,neuronthres,0)
	c+=2
'''

wt_sorted=np.sort(weight_table)[:,::-1]



for i in range(wt_sorted.shape[1]):
	if (sum(wt_sorted[:,i]<0.3)==800):
		wt_new=wt_sorted[:,:i]
		#print(wt_new.shape)
		break

'''
#plt.figure(figsize=(10, 8),dpi=80)
plt.figure(figsize=(40, 40),dpi=180)
ax = sns.heatmap(wt_new, cmap="inferno", vmin=0.3 , vmax = np.max(wt_sorted))
#ax.get_xaxis().set_ticks([])
ax.set_xticklabels(range(1,wt_new.shape[1]+1),fontsize=20,fontweight='bold')
ax.get_yaxis().set_ticks(range(5,800,10))
ax.set_yticklabels(range(1,81), rotation = 0,fontsize=20,fontweight='bold')
ax.set_xlabel('SYNAPSE',fontsize=30,fontweight='bold')
ax.set_ylabel('NEURON',fontsize=30,fontweight='bold')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=30)
plt.title('WEIGHT INTENSITY PER SYNAPSE', fontweight= "bold",fontsize=35,loc='center')

#for i in range(1,wt_sorted.shape[1]+1):
#    ax.axvline(i, color='white', lw=1)
#ax.axvline(2, color='blue', lw=5)
#plt.show()
plt.savefig("./temp/Synapses_Heatmap",bbox_inches='tight')
'''


str_syns_per_branch=np.sum((wt_new>0.3),axis=1).reshape((80,10))

tot_syns=np.sum((wt_new>0),axis=1)

tot_syns.shape

cluster_branch_per_neuron=np.sum((str_syns_per_branch>2),axis=1)


sum(cluster_branch_per_neuron[:40]>0)
sum(cluster_branch_per_neuron[40:]>0)

np.sum(str_syns_per_branch[:40])
np.sum(str_syns_per_branch[40:])

print(np.sum(tot_syns[:400]),np.sum(tot_syns[400:]))

plt.figure(figsize=(8,8),dpi=80)
ax=sns.barplot([1,2],[np.sum(tot_syns[:400]),np.sum(tot_syns[400:])])
ax.set_xlabel('Subpopulation',fontsize=12,fontweight='bold')
ax.set_ylabel('Number of Synapses',fontsize=12,fontweight='bold')
ax.set_ylim(0,1250)
plt.title('TOTAL SYNAPSES PER SUBPOPULATION', fontweight= "bold",fontsize=15,loc='center')
plt.show()

'''
plt.figure(figsize=(8,8),dpi=80)
ax=sns.barplot([1,2],[np.sum(str_syns_per_branch[:40]),np.sum(str_syns_per_branch[40:])])
ax.set_xlabel('Subpopulation',fontsize=12,fontweight='bold')
ax.set_ylabel('Number of Synapses',fontsize=12,fontweight='bold')
ax.set_ylim(0,550)
plt.title('EFFECTIVE SYNAPSES(w>0.3) PER SUBPOPULATION', fontweight= "bold",fontsize=15,loc='center')
plt.show()
#plt.savefig("./temp/Effective_syns",bbox_inches='tight')

plt.figure(figsize=(8,8),dpi=80)
ax=sns.barplot([1,2],[sum(cluster_branch_per_neuron[:40]>0),sum(cluster_branch_per_neuron[40:]>0)])
ax.set_xlabel('Subpopulation',fontsize=12,fontweight='bold')
ax.set_ylabel('Number of Neurons',fontsize=12,fontweight='bold')
ax.set_ylim(0,55)
plt.title('Number of Neurons with at least 1 cluster', fontweight= "bold",fontsize=15,loc='center')
#plt.show()
plt.savefig("./temp/Neurons_with_1_cluster",bbox_inches='tight')

plt.figure(figsize=(8,8),dpi=80)
ax=sns.barplot([1,2],[sum(cluster_branch_per_neuron[:40]>1),sum(cluster_branch_per_neuron[40:]>1)])
ax.set_xlabel('Subpopulation',fontsize=12,fontweight='bold')
ax.set_ylabel('Number of Neurons',fontsize=12,fontweight='bold')
ax.set_ylim(0,55)
plt.title('Number of Neurons with at least 2 clusters', fontweight= "bold",fontsize=15,loc='center')
#plt.show()
plt.savefig("./temp/Neurons_with_2_clusters",bbox_inches='tight')
'''
'''
plt.figure(figsize=(8,8),dpi=80)
ax=sns.barplot([1,2],[sum(cluster_branch_per_neuron[:40]>2),sum(cluster_branch_per_neuron[40:]>2)])
ax.set_xlabel('Subpopulation',fontsize=12,fontweight='bold')
ax.set_ylabel('Number of Neurons',fontsize=12,fontweight='bold')
ax.set_ylim(0,55)
plt.title('Number of Neurons with at least 3 clusters', fontweight= "bold",fontsize=15,loc='center')
plt.show()

plt.figure(figsize=(8,8),dpi=80)
ax=sns.barplot([1,2],[sum(cluster_branch_per_neuron[:40]>3),sum(cluster_branch_per_neuron[40:]>3)])
ax.set_xlabel('Subpopulation',fontsize=12,fontweight='bold')
ax.set_ylabel('Number of Neurons',fontsize=12,fontweight='bold')
ax.set_ylim(0,55)
plt.title('Number of Neurons with at least 4 clusters', fontweight= "bold",fontsize=15,loc='center')
plt.show()

plt.figure(figsize=(8,8),dpi=80)
ax=sns.barplot([1,2],[sum(cluster_branch_per_neuron[:40]>4),sum(cluster_branch_per_neuron[40:]>4)])
ax.set_xlabel('Subpopulation',fontsize=12,fontweight='bold')
ax.set_ylabel('Number of Neurons',fontsize=12,fontweight='bold')
ax.set_ylim(0,55)
plt.title('Number of Neurons with at least 5 clusters', fontweight= "bold",fontsize=15,loc='center')
plt.show()
'''
