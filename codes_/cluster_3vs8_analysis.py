import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
from scipy.stats import pearsonr



full_path_data ="/media/nikos/Elements/ClusterResults/results_181121_3vs8_200runs_8280interstim_constrained_turnover_alternating_chunks_3/"
suffix = "_nikos_"

save_figs=False

digit1=str(3)
digit2=str(8)


bug_seed_list=[]

file_type ="sample_accuracies"
plt.figure(figsize=(10, 10), dpi=80) ##FOR LEARNING CURVES
	
for s in range(1,201):
	if s not in bug_seed_list:
		seed = str(s)
		df=pd.read_table(full_path_data + file_type + suffix + seed + "_digits_" 
	      + digit1 + "_" + digit2 + ".txt",header=None, error_bad_lines=False, sep = " ")	
	
		y=np.vstack(([[0]],df.values))
		x = list(range(30*0,30*len(y),30))			
		plt.plot(x,y)
		plt.grid(True)
		plt.xlabel('Training Iterations', fontweight = 'bold')
		plt.yticks(np.arange(0,110,10))
		plt.ylabel('% Accuracy', fontweight = 'bold')
		plt.title('Learning Curves ' + digit1 + " vs " + digit2, fontsize=14, fontweight='bold')

if save_figs:
	plt.savefig("./temp/3vs8AllLearningCurves.png")
else:
	plt.show()


file_type ="predictons"
accs=[]

####################GRABING GOOD AND BAD SEEDS#######################

for s in range(1,201):
	if s not in bug_seed_list:
		seed = str(s)
		df=pd.read_table(full_path_data + file_type + suffix + seed + "_digits_" 
	      + digit1 + "_" + digit2 + ".txt",header=None, error_bad_lines=False, sep = " ")	
	
		acc=round(sum(df[2])/len(df[2]), 2)
		accs.append(acc)


accs_arr=np.array(accs)



sum(accs_arr<=0.75)
sum(accs_arr>=0.85)
np.mean(accs_arr)
np.std(accs_arr)
np.max(accs_arr)
np.min(accs_arr)


#Accuracy Distribution plot
plt.figure(figsize=(10, 10), dpi=80)
plt.hist(accs_arr,bins=20,ec="k",color='plum')
plt.ylim(0,60)
plt.xlim(0.4,1.)
plt.xlabel('Accuracy Value',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.title('Accuracy Distribution',fontweight='bold',fontsize=15)
if save_figs:
	plt.savefig("./temp/AccuracyDistribution.png")
else:
	plt.show()
	
	

good_performance_seeds=np.where(accs_arr>=0.85)[0]+1

#MANUALLY FIX IF BUG SEED, IF NO BUG SEED, COMMENT OUT:
#good_performance_seeds[good_performance_seeds>=26]+=1

bad_performance_seeds=np.where(accs_arr<=0.74)[0]+1

#MANUALLY FIX IF BUG SEED, IF NO BUG SEED, COMMENT OUT:
#bad_performance_seeds[bad_performance_seeds>=26]+=1

file_type ="sample_accuracies"

plt.figure(figsize=(10, 10), dpi=80) ##FOR GOOD LEARNING CURVES
	
for s in good_performance_seeds:
	if s not in bug_seed_list:
		seed = str(s)
		df=pd.read_table(full_path_data + file_type + suffix + seed + "_digits_" 
	      + digit1 + "_" + digit2 + ".txt",header=None, error_bad_lines=False, sep = " ")	
	
		y=np.vstack(([[0]],df.values))
		x = list(range(30*0,30*len(y),30))			
		plt.plot(x,y)
		plt.grid(True)
		plt.xlabel('Training Iterations', fontweight = 'bold')
		plt.yticks(np.arange(0,110,10))
		plt.ylabel('% Accuracy', fontweight = 'bold')
		plt.title('Good Learning Curves ' + digit1 + " vs " + digit2, fontsize=14, fontweight='bold')
		
if save_figs:
	plt.savefig("./temp/3vs8GoodLearningCurves.png")
else:
	plt.show()

plt.figure(figsize=(10, 10), dpi=80) ##FOR BAD LEARNING CURVES
	
for s in bad_performance_seeds:
	if s not in bug_seed_list:
		seed = str(s)
		df=pd.read_table(full_path_data + file_type + suffix + seed + "_digits_" 
	      + digit1 + "_" + digit2 + ".txt",header=None, error_bad_lines=False, sep = " ")	
	
		y=np.vstack(([[0]],df.values))
		x = list(range(30*0,30*len(y),30))			
		plt.plot(x,y)
		plt.grid(True)
		plt.xlabel('Training Iterations', fontweight = 'bold')
		plt.yticks(np.arange(0,110,10))
		plt.ylabel('% Accuracy', fontweight = 'bold')
		plt.title('Bad Learning Curves ' + digit1 + " vs " + digit2, fontsize=14, fontweight='bold')
		
if save_figs:
	plt.savefig("./temp/3vs8BadLearningCurves.png")
else:
	plt.show()


file_type ="synapse_analysis"


tot_syns_3_list=[]
tot_syns_8_list=[]

str_syns_3_list=[]
str_syns_8_list=[]

neurons_with_1_clusters_3_list=[]
neurons_with_2_clusters_3_list=[]
neurons_with_3_clusters_3_list=[]
neurons_with_4_clusters_3_list=[]
neurons_with_5_clusters_3_list=[]

neurons_with_1_clusters_8_list=[]
neurons_with_2_clusters_8_list=[]
neurons_with_3_clusters_8_list=[]
neurons_with_4_clusters_8_list=[]
neurons_with_5_clusters_8_list=[]


for s in good_performance_seeds:
	if s not in bug_seed_list:
		seed = str(s)
		data_file=full_path_data + file_type + suffix + seed + "_digits_" + digit1 + "_" + digit2 + ".txt"				

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
		wt_sorted=np.sort(weight_table)[:,::-1]

		for i in range(wt_sorted.shape[1]):
			if (sum(wt_sorted[:,i]<0.3)==800):
				wt_new=wt_sorted[:,:i]
				#print(wt_new.shape)
				break
		
				
		str_syns_per_branch=np.sum((wt_new>0.3),axis=1).reshape((80,10))
		
		tot_syns=np.sum((wt_new>0),axis=1)
		
		cluster_branch_per_neuron=np.sum((str_syns_per_branch>2),axis=1)
		
		
		tot_syns_3=np.sum(tot_syns[:400])
		tot_syns_3_list.append(tot_syns_3)
		
		tot_syns_8=np.sum(tot_syns[400:])
		tot_syns_8_list.append(tot_syns_8)
		
		str_syns_3=np.sum(str_syns_per_branch[:40])
		str_syns_3_list.append(str_syns_3)

		str_syns_8=np.sum(str_syns_per_branch[40:])
		str_syns_8_list.append(str_syns_8)
		
		neurons_with_1_clusters_3=sum(cluster_branch_per_neuron[:40]>0)
		neurons_with_1_clusters_3_list.append(neurons_with_1_clusters_3)
		
		neurons_with_2_clusters_3=sum(cluster_branch_per_neuron[:40]>1)
		neurons_with_2_clusters_3_list.append(neurons_with_2_clusters_3)
		
		neurons_with_3_clusters_3=sum(cluster_branch_per_neuron[:40]>2)
		neurons_with_3_clusters_3_list.append(neurons_with_3_clusters_3)
		
		neurons_with_4_clusters_3=sum(cluster_branch_per_neuron[:40]>3)
		neurons_with_4_clusters_3_list.append(neurons_with_4_clusters_3)
		
		neurons_with_5_clusters_3=sum(cluster_branch_per_neuron[:40]>4)
		neurons_with_5_clusters_3_list.append(neurons_with_5_clusters_3)
		
		
		neurons_with_1_clusters_8=sum(cluster_branch_per_neuron[40:]>0)
		neurons_with_1_clusters_8_list.append(neurons_with_1_clusters_8)
		
		neurons_with_2_clusters_8=sum(cluster_branch_per_neuron[40:]>1)
		neurons_with_2_clusters_8_list.append(neurons_with_2_clusters_8)
		
		neurons_with_3_clusters_8=sum(cluster_branch_per_neuron[40:]>2)
		neurons_with_3_clusters_8_list.append(neurons_with_3_clusters_8)
		
		neurons_with_4_clusters_8=sum(cluster_branch_per_neuron[40:]>3)
		neurons_with_4_clusters_8_list.append(neurons_with_4_clusters_8)
		
		neurons_with_5_clusters_8=sum(cluster_branch_per_neuron[40:]>4)
		neurons_with_5_clusters_8_list.append(neurons_with_5_clusters_8)


		
good_tot_syns_diff=np.abs(np.array(tot_syns_3_list)-np.array(tot_syns_8_list))
good_str_syns_diff=np.abs(np.array(str_syns_3_list)-np.array(str_syns_8_list))
good_neurons_with_1_cluster_diff=np.abs(np.array(neurons_with_1_clusters_3_list)-np.array(neurons_with_1_clusters_8_list))
good_neurons_with_2_cluster_diff=np.abs(np.array(neurons_with_2_clusters_3_list)-np.array(neurons_with_2_clusters_8_list))
good_neurons_with_3_cluster_diff=np.abs(np.array(neurons_with_3_clusters_3_list)-np.array(neurons_with_3_clusters_8_list))
good_neurons_with_4_cluster_diff=np.abs(np.array(neurons_with_4_clusters_3_list)-np.array(neurons_with_4_clusters_8_list))
good_neurons_with_5_cluster_diff=np.abs(np.array(neurons_with_5_clusters_3_list)-np.array(neurons_with_5_clusters_8_list))


#PLOTS FOR GOOD SEEDS
plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(tot_syns_3_list),np.mean(tot_syns_8_list)],
		width=0.4, yerr=[np.std(tot_syns_3_list),np.std(tot_syns_8_list)],
		edgecolor ='black',color='palegoldenrod')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,1200)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Total Synapses per Subpopulation', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/TotSynsGood.png")
else:
	plt.show()




plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(str_syns_3_list),np.mean(str_syns_8_list)],
		width=0.4, yerr=[np.std(str_syns_3_list),np.std(str_syns_8_list)],
		edgecolor ='black',color='palegoldenrod')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,1200)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Effective Synapses(w>0.3) per Subpopulation', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/StrSynsGood.png")
else:
	plt.show()




plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(neurons_with_1_clusters_3_list),np.mean(neurons_with_1_clusters_8_list)],
		width=0.4, yerr=[np.std(neurons_with_1_clusters_3_list),np.std(neurons_with_1_clusters_8_list)],
		edgecolor ='black',color='maroon')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,45)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Number of Neurons with at least 1 cluster', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith1ClusterGood.png")
else:
	plt.show()





plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(neurons_with_2_clusters_3_list),np.mean(neurons_with_2_clusters_8_list)],
		width=0.4, yerr=[np.std(neurons_with_2_clusters_3_list),np.std(neurons_with_2_clusters_8_list)],
		edgecolor ='black',color='maroon')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,40)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Number of Neurons with at least 2 clusters', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith2ClusterGood.png")
else:
	plt.show()




plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(neurons_with_3_clusters_3_list),np.mean(neurons_with_3_clusters_8_list)],
		width=0.4, yerr=[np.std(neurons_with_3_clusters_3_list),np.std(neurons_with_3_clusters_8_list)],
		edgecolor ='black',color='maroon')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,40)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Number of Neurons with at least 3 clusters', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith3ClusterGood.png")
else:
	plt.show()





plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(neurons_with_4_clusters_3_list),np.mean(neurons_with_4_clusters_8_list)],
		width=0.4, yerr=[np.std(neurons_with_4_clusters_3_list),np.std(neurons_with_4_clusters_8_list)],
		edgecolor ='black',color='maroon')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,40)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Number of Neurons with at least 4 clusters', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith4ClusterGood.png")
else:
	plt.show()





plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(neurons_with_5_clusters_3_list),np.mean(neurons_with_5_clusters_8_list)],
		width=0.4, yerr=[np.std(neurons_with_5_clusters_3_list),np.std(neurons_with_5_clusters_8_list)],
		edgecolor ='black',color='maroon')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,40)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Number of Neurons with at least 5 clusters', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith5ClusterGood.png")
else:
	plt.show()


















tot_syns_3_list=[]
tot_syns_8_list=[]

str_syns_3_list=[]
str_syns_8_list=[]

neurons_with_1_clusters_3_list=[]
neurons_with_2_clusters_3_list=[]
neurons_with_3_clusters_3_list=[]
neurons_with_4_clusters_3_list=[]
neurons_with_5_clusters_3_list=[]

neurons_with_1_clusters_8_list=[]
neurons_with_2_clusters_8_list=[]
neurons_with_3_clusters_8_list=[]
neurons_with_4_clusters_8_list=[]
neurons_with_5_clusters_8_list=[]


for s in bad_performance_seeds:
	if s not in bug_seed_list:
		seed = str(s)
		data_file=full_path_data + file_type + suffix + seed + "_digits_" + digit1 + "_" + digit2 + ".txt"				

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
		wt_sorted=np.sort(weight_table)[:,::-1]

		for i in range(wt_sorted.shape[1]):
			if (sum(wt_sorted[:,i]<0.3)==800):
				wt_new=wt_sorted[:,:i]
				#print(wt_new.shape)
				break
		
				
		str_syns_per_branch=np.sum((wt_new>0.3),axis=1).reshape((80,10))
		
		tot_syns=np.sum((wt_new>0),axis=1)
		
		cluster_branch_per_neuron=np.sum((str_syns_per_branch>2),axis=1)
		
		
		tot_syns_3=np.sum(tot_syns[:400])
		tot_syns_3_list.append(tot_syns_3)
		
		tot_syns_8=np.sum(tot_syns[400:])
		tot_syns_8_list.append(tot_syns_8)
		
		str_syns_3=np.sum(str_syns_per_branch[:40])
		str_syns_3_list.append(str_syns_3)

		str_syns_8=np.sum(str_syns_per_branch[40:])
		str_syns_8_list.append(str_syns_8)
		
		neurons_with_1_clusters_3=sum(cluster_branch_per_neuron[:40]>0)
		neurons_with_1_clusters_3_list.append(neurons_with_1_clusters_3)
		
		neurons_with_2_clusters_3=sum(cluster_branch_per_neuron[:40]>1)
		neurons_with_2_clusters_3_list.append(neurons_with_2_clusters_3)
		
		neurons_with_3_clusters_3=sum(cluster_branch_per_neuron[:40]>2)
		neurons_with_3_clusters_3_list.append(neurons_with_3_clusters_3)
		
		neurons_with_4_clusters_3=sum(cluster_branch_per_neuron[:40]>3)
		neurons_with_4_clusters_3_list.append(neurons_with_4_clusters_3)
		
		neurons_with_5_clusters_3=sum(cluster_branch_per_neuron[:40]>4)
		neurons_with_5_clusters_3_list.append(neurons_with_5_clusters_3)
		
		
		neurons_with_1_clusters_8=sum(cluster_branch_per_neuron[40:]>0)
		neurons_with_1_clusters_8_list.append(neurons_with_1_clusters_8)
		
		neurons_with_2_clusters_8=sum(cluster_branch_per_neuron[40:]>1)
		neurons_with_2_clusters_8_list.append(neurons_with_2_clusters_8)
		
		neurons_with_3_clusters_8=sum(cluster_branch_per_neuron[40:]>2)
		neurons_with_3_clusters_8_list.append(neurons_with_3_clusters_8)
		
		neurons_with_4_clusters_8=sum(cluster_branch_per_neuron[40:]>3)
		neurons_with_4_clusters_8_list.append(neurons_with_4_clusters_8)
		
		neurons_with_5_clusters_8=sum(cluster_branch_per_neuron[40:]>4)
		neurons_with_5_clusters_8_list.append(neurons_with_5_clusters_8)

		
bad_tot_syns_diff=np.abs(np.array(tot_syns_3_list)-np.array(tot_syns_8_list))
bad_str_syns_diff=np.abs(np.array(str_syns_3_list)-np.array(str_syns_8_list))
bad_neurons_with_1_cluster_diff=np.abs(np.array(neurons_with_1_clusters_3_list)-np.array(neurons_with_1_clusters_8_list))
bad_neurons_with_2_cluster_diff=np.abs(np.array(neurons_with_2_clusters_3_list)-np.array(neurons_with_2_clusters_8_list))
bad_neurons_with_3_cluster_diff=np.abs(np.array(neurons_with_3_clusters_3_list)-np.array(neurons_with_3_clusters_8_list))
bad_neurons_with_4_cluster_diff=np.abs(np.array(neurons_with_4_clusters_3_list)-np.array(neurons_with_4_clusters_8_list))
bad_neurons_with_5_cluster_diff=np.abs(np.array(neurons_with_5_clusters_3_list)-np.array(neurons_with_5_clusters_8_list))



#PLOTS FOR BAD SEEDS
plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(tot_syns_3_list),np.mean(tot_syns_8_list)],
		width=0.4, yerr=[np.std(tot_syns_3_list),np.std(tot_syns_8_list)],
		edgecolor ='black',color='palegoldenrod')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,1200)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Total Synapses per Subpopulation', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/TotSynsBad.png")
else:
	plt.show()






plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(str_syns_3_list),np.mean(str_syns_8_list)],
		width=0.4, yerr=[np.std(str_syns_3_list),np.std(str_syns_8_list)],
		edgecolor ='black',color='palegoldenrod')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,1200)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Effective Synapses(w>0.3) per Subpopulation', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/StrSynsBad.png")
else:
	plt.show()






plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(neurons_with_1_clusters_3_list),np.mean(neurons_with_1_clusters_8_list)],
		width=0.4, yerr=[np.std(neurons_with_1_clusters_3_list),np.std(neurons_with_1_clusters_8_list)],
		edgecolor ='black',color='maroon')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,45)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Number of Neurons with at least 1 cluster', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith1ClusterBad.png")
else:
	plt.show()





plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(neurons_with_2_clusters_3_list),np.mean(neurons_with_2_clusters_8_list)],
		width=0.4, yerr=[np.std(neurons_with_2_clusters_3_list),np.std(neurons_with_2_clusters_8_list)],
		edgecolor ='black',color='maroon')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,40)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Number of Neurons with at least 2 clusters', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith2ClusterBad.png")
else:
	plt.show()





plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(neurons_with_3_clusters_3_list),np.mean(neurons_with_3_clusters_8_list)],
		width=0.4, yerr=[np.std(neurons_with_3_clusters_3_list),np.std(neurons_with_3_clusters_8_list)],
		edgecolor ='black',color='maroon')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,40)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Number of Neurons with at least 3 clusters', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith3ClusterBad.png")
else:
	plt.show()






plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(neurons_with_4_clusters_3_list),np.mean(neurons_with_4_clusters_8_list)],
		width=0.4, yerr=[np.std(neurons_with_4_clusters_3_list),np.std(neurons_with_4_clusters_8_list)],
		edgecolor ='black',color='maroon')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,40)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Number of Neurons with at least 4 clusters', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith4ClusterBad.png")
else:
	plt.show()






plt.figure(figsize=(8,8),dpi=80)
plt.bar([0,1],
		[np.mean(neurons_with_5_clusters_3_list),np.mean(neurons_with_5_clusters_8_list)],
		width=0.4, yerr=[np.std(neurons_with_5_clusters_3_list),np.std(neurons_with_5_clusters_8_list)],
		edgecolor ='black',color='maroon')
plt.xticks([0,1],['Subpopulation 3','Subpopulation 8'],fontweight='bold',fontsize='12')
plt.ylim(0,40)
plt.ylabel('Number of Synapses',fontsize=12,fontweight='bold')
plt.title('Number of Neurons with at least 5 clusters', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith5ClusterBad.png")
else:
	plt.show()





#COMPARATIVE PLOTS FOR ABSOLUTE DIFFERENCE IN METRICS

plt.figure(figsize=(12,12),dpi=80)
plt.bar([0,1],
		[np.mean(good_tot_syns_diff),np.mean(bad_tot_syns_diff)],
		width=0.4, yerr=[np.std(good_tot_syns_diff),np.std(bad_tot_syns_diff)],
		edgecolor ='black',color='aquamarine')
plt.xticks([0,1],['Good Performance','Bad Performance'],fontweight='bold',fontsize='12')
plt.ylim(0,300)
plt.ylabel('Absolute Difference(Synapses)',fontsize=12,fontweight='bold')
plt.title('Absolute Difference in Total Synapses between Subpopulations', fontweight= "bold",fontsize=14,loc='center')

if save_figs:
	plt.savefig("./temp/TotSynsDiff.png")
else:
	plt.show()





plt.figure(figsize=(12,12),dpi=80)
plt.bar([0,1],
		[np.mean(good_str_syns_diff),np.mean(bad_str_syns_diff)],
		width=0.4, yerr=[np.std(good_str_syns_diff),np.std(bad_str_syns_diff)],
		edgecolor ='black',color='aquamarine')
plt.xticks([0,1],['Good Performance','Bad Performance'],fontweight='bold',fontsize='12')
plt.ylim(0,300)
plt.ylabel('Absolute Difference(Synapses)',fontsize=12,fontweight='bold')
plt.title('Absolute Difference in Effective Synapses(w>0.3) between Subpopulations', fontweight= "bold",fontsize=14,loc='center')

if save_figs:
	plt.savefig("./temp/StrSynsDiff.png")
else:
	plt.show()





plt.figure(figsize=(12,12),dpi=80)
plt.bar([0,1],
		[np.mean(good_neurons_with_1_cluster_diff),np.mean(bad_neurons_with_1_cluster_diff)],
		width=0.4, yerr=[np.std(good_neurons_with_1_cluster_diff),np.std(bad_neurons_with_1_cluster_diff)],
		edgecolor ='black',color='salmon')
plt.xticks([0,1],['Good Performance','Bad Performance'],fontweight='bold',fontsize='12')
plt.ylim(0,20)
plt.ylabel('Absolute Difference(Neurons)',fontsize=12,fontweight='bold')
plt.title('Absolute Difference in Number of Neurons with at least 1 cluster', fontweight= "bold",fontsize=14,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith1ClusterDiff.png")
else:
	plt.show()





plt.figure(figsize=(12,12),dpi=80)
plt.bar([0,1],
		[np.mean(good_neurons_with_2_cluster_diff),np.mean(bad_neurons_with_2_cluster_diff)],
		width=0.4, yerr=[np.std(good_neurons_with_2_cluster_diff),np.std(bad_neurons_with_2_cluster_diff)],
		edgecolor ='black',color='salmon')
plt.xticks([0,1],['Good Performance','Bad Performance'],fontweight='bold',fontsize='12')
plt.ylim(0,20)
plt.ylabel('Absolute Difference(Neurons)',fontsize=12,fontweight='bold')
plt.title('Absolute Difference in Number of Neurons with at least 2 clusters', fontweight= "bold",fontsize=14,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith2ClusterDiff.png")
else:
	plt.show()





plt.figure(figsize=(12,12),dpi=80)
plt.bar([0,1],
		[np.mean(good_neurons_with_3_cluster_diff),np.mean(bad_neurons_with_3_cluster_diff)],
		width=0.4, yerr=[np.std(good_neurons_with_3_cluster_diff),np.std(bad_neurons_with_3_cluster_diff)],
		edgecolor ='black',color='salmon')
plt.xticks([0,1],['Good Performance','Bad Performance'],fontweight='bold',fontsize='12')
plt.ylim(0,20)
plt.ylabel('Absolute Difference(Neurons)',fontsize=12,fontweight='bold')
plt.title('Absolute Difference in Number of Neurons with at least 3 clusters', fontweight= "bold",fontsize=14,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith3ClusterDiff.png")
else:
	plt.show()
	




plt.figure(figsize=(12,12),dpi=80)
plt.bar([0,1],
		[np.mean(good_neurons_with_4_cluster_diff),np.mean(bad_neurons_with_4_cluster_diff)],
		width=0.4, yerr=[np.std(good_neurons_with_4_cluster_diff),np.std(bad_neurons_with_4_cluster_diff)],
		edgecolor ='black',color='salmon')
plt.xticks([0,1],['Good Performance','Bad Performance'],fontweight='bold',fontsize='12')
plt.ylim(0,20)
plt.ylabel('Absolute Difference(Neurons)',fontsize=12,fontweight='bold')
plt.title('Absolute Difference in Number of Neurons with at least 4 clusters', fontweight= "bold",fontsize=14,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith4ClusterDiff.png")
else:
	plt.show()
	




plt.figure(figsize=(12,12),dpi=80)
plt.bar([0,1],
		[np.mean(good_neurons_with_5_cluster_diff),np.mean(bad_neurons_with_5_cluster_diff)],
		width=0.4, yerr=[np.std(good_neurons_with_5_cluster_diff),np.std(bad_neurons_with_5_cluster_diff)],
		edgecolor ='black',color='salmon')
plt.xticks([0,1],['Good Performance','Bad Performance'],fontweight='bold',fontsize='12')
plt.ylim(0,20)
plt.ylabel('Absolute Difference(Neurons)',fontsize=12,fontweight='bold')
plt.title('Absolute Difference in Number of Neurons with at least 5 clusters', fontweight= "bold",fontsize=14,loc='center')

if save_figs:
	plt.savefig("./temp/NeuronsWith5ClusterDiff.png")
else:
	plt.show()
	



#Activity analysis for overlap. Takes some time to run, comment out if not needed!!!
'''
file_type ="test_activity"
overlaps=[]
overlaps_sub1=[]
overlaps_sub2=[]
for s in range(1,201):
	if s not in bug_seed_list:
		seed = str(s)
		df=pd.read_table(full_path_data + file_type + suffix + seed + "_digits_" 
	      + digit1 + "_" + digit2 + ".txt")	
		

		dfd1=df[df['LABEL']==digit1]
		spikesumd1=np.zeros(800)
		c=0
		div=0
		
		for i in range(800,dfd1.shape[0],800):
			div+=1
			temp_dfd1=dfd1.iloc()[c:i]
			spikesumd1+=temp_dfd1['NEURON_SPIKES'].values.astype(np.int)
			c=i
		
		spikesumd1/=div
		
		dfd2=df[df['LABEL']==digit2]
		spikesumd2=np.zeros(800)
		c=0
		div=0
		
		for i in range(800,dfd2.shape[0],800):
			div+=1
			temp_dfd2=dfd2.iloc()[c:i]
			spikesumd2+=temp_dfd2['NEURON_SPIKES'].values.astype(np.int)
			c=i
		
		spikesumd2/=div
		
		
		neuronspikesd1=np.zeros(80)
		neuronspikesd2=np.zeros(80)
		j=0
		
		for i in range(0,len(spikesumd1),10):
			neuronspikesd1[j]=round(spikesumd1[i])
			neuronspikesd2[j]=round(spikesumd2[i])
			j+=1
		
		
		
		
		neuronspikesd1sub1=neuronspikesd1[:40].reshape(40,1)
		neuronspikesd1sub2=neuronspikesd1[40:].reshape(40,1)
		neuronspikesd2sub1=neuronspikesd2[:40].reshape(40,1)
		neuronspikesd2sub2=neuronspikesd2[40:].reshape(40,1)
		
		
		meanneuronspikesperdigit=np.vstack((neuronspikesd1,neuronspikesd2)).T
		
		overlap_perc_tot = 100*np.sum(np.sum(meanneuronspikesperdigit>20,axis=1)==2)/len(meanneuronspikesperdigit)
		overlaps.append(overlap_perc_tot)
		
		overlap_perc_sub1 = 100*np.sum(np.sum(meanneuronspikesperdigit[:40]>20,axis=1)==2)/len(meanneuronspikesperdigit[:40])
		overlaps_sub1.append(overlap_perc_sub1)
		
		overlap_perc_sub2 = 100*np.sum(np.sum(meanneuronspikesperdigit[40:]>20,axis=1)==2)/len(meanneuronspikesperdigit[40:])
		overlaps_sub2.append(overlap_perc_sub2)



overlaps_arr=np.array(overlaps)


a, b = np.polyfit(overlaps_arr, accs_arr, deg=1)
y_est = a * overlaps_arr + b
R,_=pearsonr(overlaps,accs)


plt.figure(figsize=(10,8),dpi=80)
plt.plot(overlaps_arr,y_est,c='r')
plt.scatter(overlaps_arr,accs_arr,c="black")
plt.text(16,0.50,'Pearson R = '+str(round(R,2)),fontweight='bold',fontsize=12)
plt.xlabel('%Overlap',fontsize=12,fontweight='bold')
plt.ylabel('Accuracy',fontsize=12,fontweight='bold')
plt.title('Overlap to accuracy plot(200 simulations)', fontweight= "bold",fontsize=15,loc='center')

if save_figs:
	plt.savefig("./temp/Acc_to_overlap_plot.png")
else:
	plt.show()

'''


plt.close("all")