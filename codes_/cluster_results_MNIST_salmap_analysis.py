import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle
from scipy.spatial import distance
from mnist import MNIST




#full_path_data ="/media/nikos/Elements/ClusterResults/results_121122_MNIST_synapses_experiment/"
#full_path_data ="/media/nikos/Elements/ClusterResults/results_160922_3500_syns_early_stop/"
full_path_data ="/media/nikos/Elements/ClusterResults/results_190922_MNIST_linear_net/"

suffix = "_nikos_"
bad_seeds=[]



mndata=MNIST('./MNIST/')
imagelists,labels=mndata.load_testing()
labels=np.array(labels)


dim=784

turns=[0,2]
digits=[0,1,2,3,4,5,6,7,8,9]
syns=[3500]
runs =list(range(1,21))
pairs=[]
for digit_2 in digits:
	for digit_1 in range(digit_2):
		pairs.append((digit_1,digit_2))

pairs=[(1,4),(3,6),(2,5),(3,8),(5,8),(4,9)] #difficulty increase per 2


#SALIENCY MAP DISTANCE ANALYSIS
syn=1750


mean_dist_sal_11_list=[]
mean_dist_sal_12_list=[]
mean_dist_sal_21_list=[]
mean_dist_sal_22_list=[]

std_dist_sal_11_list=[]
std_dist_sal_12_list=[]
std_dist_sal_21_list=[]
std_dist_sal_22_list=[]

indexlist=[]


for turn in turns:
	turn_dists_mean_11=[]
	turn_dists_mean_12=[]
	turn_dists_mean_21=[]
	turn_dists_mean_22=[]
	
	turn_dists_std_11=[]
	turn_dists_std_12=[]
	turn_dists_std_21=[]
	turn_dists_std_22=[]
	
	for (digit_1,digit_2) in pairs:

			
		runs_dists_mean_11=[]
		runs_dists_mean_12=[]
		runs_dists_mean_21=[]
		runs_dists_mean_22=[]
		
		runs_dists_std_11=[]
		runs_dists_std_12=[]
		runs_dists_std_21=[]
		runs_dists_std_22=[]
		
		
		
		class_1=digit_1
		class_2=digit_2
		
		cls1_indexes = np.where(labels==class_1)[0]
		cls2_indexes = np.where(labels==class_2)[0]
		x=np.hstack((cls1_indexes,cls2_indexes))
		x.sort()
		testset=np.array(imagelists)[x]
		testset_labels=labels[x]
		
		cumweights1=[]
		cumweights2=[]
		for run in runs:
			
			
			testset_norm=testset/np.max(testset)
			
			
			testset_norm_cls_1=testset_norm[testset_labels==class_1]
			testset_norm_cls_2=testset_norm[testset_labels==class_2]
			
			testset_norm_mean=np.mean(testset_norm,axis=0)
			testset_norm_cls_1_mean=np.mean(testset_norm_cls_1,axis=0)
			testset_norm_cls_2_mean=np.mean(testset_norm_cls_2,axis=0)
			
			
			
			
			
			
			
			# Inputs
			file_type_1 ="presyn_w_sub1"
			file_type_2 ="presyn_w_sub2"
			
			# Delimiter
			data_file_delimiter = '\t'
			
			
			data_file_w=full_path_data + file_type_1 + suffix + str(run) + "_" + str(turn) + "_" + str(syn) + "_digits_" + str(class_1) +"_" +str(class_2) + ".txt"
			# The max column count a line in the file could have
			largest_column_count = 0
			
			# Loop the data lines
			with open(data_file_w, 'r') as temp_f:
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
			df_w_tot = pd.read_csv(data_file_w, header=None, delimiter=data_file_delimiter, names=column_names, error_bad_lines=False, engine='python')
			df_w_tot=df_w_tot.fillna(0)
			
			
			data_file_w2=full_path_data + file_type_2 + suffix + str(run) + "_" + str(turn) + "_" + str(syn) + "_digits_" + str(class_1) +"_" +str(class_2) + ".txt"
			# The max column count a line in the file could have
			largest_column_count = 0
			
			# Loop the data lines
			with open(data_file_w2, 'r') as temp_f:
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
			df_w2_tot = pd.read_csv(data_file_w2, header=None, delimiter=data_file_delimiter, names=column_names, error_bad_lines=False, engine='python')
			df_w2_tot=df_w2_tot.fillna(0)
			
			
			
			inputs=784
			
			df_w=df_w_tot[len(df_w_tot)-inputs:len(df_w_tot)]
			
			df_w2=df_w2_tot[len(df_w2_tot)-inputs:len(df_w2_tot)]
			
			input_ids=df_w[0]
			
			w_tab_1=df_w.values[:,1:]
			w_tab_2=df_w2.values[:,1:]
			
			#Effective weights
			#w_tab_1[w_tab_1<0.3]=0
			#w_tab_2[w_tab_2<0.3]=0
			
			
			cumweight1=np.sum(w_tab_1,axis=1).reshape((int(np.sqrt(inputs)),int(np.sqrt(inputs))))
			cumweight2=np.sum(w_tab_2,axis=1).reshape((int(np.sqrt(inputs)),int(np.sqrt(inputs))))
			
			cumweights1.append(cumweight1)
			cumweights2.append(cumweight2)
			
			
			cumweight1_norm = cumweight1/np.max(cumweight1)
			cumweight2_norm = cumweight2/np.max(cumweight2)
			
			
			cumweight1_norm_flat = cumweight1_norm.flatten()
			cumweight2_norm_flat = cumweight2_norm.flatten()
			
			
			
			dist_salmap_1=[]
			dist_salmap_2=[]
			
			dist_mean_1=[]
			dist_mean_2=[]
			
			for i in testset_norm:
				dist_salmap_1.append(distance.euclidean(cumweight1_norm_flat, i))
				dist_salmap_2.append(distance.euclidean(cumweight2_norm_flat, i))
			
				dist_mean_1.append(distance.euclidean(testset_norm_cls_1_mean, i))
				dist_mean_2.append(distance.euclidean(testset_norm_cls_2_mean, i))
				
				
			dist_salmap_1=np.array(dist_salmap_1)
			dist_salmap_2=np.array(dist_salmap_2)
		
			
			dist_mean_1=np.array(dist_mean_1)
			dist_mean_2=np.array(dist_mean_2)
		
			
			dist_salmap_1_from_cls_1 = dist_salmap_1[testset_labels==class_1]
			dist_salmap_1_from_cls_2 = dist_salmap_1[testset_labels==class_2]
			dist_salmap_2_from_cls_1 = dist_salmap_2[testset_labels==class_1]
			dist_salmap_2_from_cls_2 = dist_salmap_2[testset_labels==class_2]
			
			dist_mean_1_from_cls_1 = dist_mean_1[testset_labels==class_1]
			dist_mean_1_from_cls_2 = dist_mean_1[testset_labels==class_2]
			dist_mean_2_from_cls_1 = dist_mean_2[testset_labels==class_1]
			dist_mean_2_from_cls_2 = dist_mean_2[testset_labels==class_2]
			
			
			mean_dist_sal_11=np.mean(dist_salmap_1_from_cls_1)
			runs_dists_mean_11.append(mean_dist_sal_11)
			std_dist_sal_11=np.std(dist_salmap_1_from_cls_1)		
			runs_dists_std_11.append(std_dist_sal_11)
			
			mean_dist_sal_12=np.mean(dist_salmap_1_from_cls_2)
			runs_dists_mean_12.append(mean_dist_sal_12)
			std_dist_sal_12=np.std(dist_salmap_1_from_cls_2)
			runs_dists_std_12.append(std_dist_sal_12)
			
			mean_dist_sal_21=np.mean(dist_salmap_2_from_cls_1)
			runs_dists_mean_21.append(mean_dist_sal_21)
			std_dist_sal_21=np.std(dist_salmap_2_from_cls_1)
			runs_dists_std_21.append(std_dist_sal_21)
			
			mean_dist_sal_22=np.mean(dist_salmap_2_from_cls_2)
			runs_dists_mean_22.append(mean_dist_sal_22)
			std_dist_sal_22=np.std(dist_salmap_2_from_cls_2)
			runs_dists_std_22.append(std_dist_sal_22)
			
	
		
			'''
			mean_dist_mean_11=np.mean(dist_mean_1_from_cls_1)
			std_dist_mean_11=np.std(dist_mean_1_from_cls_1)
			
			mean_dist_mean_12=np.mean(dist_mean_1_from_cls_2)
			std_dist_mean_12=np.std(dist_mean_1_from_cls_2)
			
			mean_dist_mean_21=np.mean(dist_mean_2_from_cls_1)
			std_dist_mean_21=np.std(dist_mean_2_from_cls_1)
			
			mean_dist_mean_22=np.mean(dist_mean_2_from_cls_2)
			std_dist_mean_22=np.std(dist_mean_2_from_cls_2)
			'''
			
			
			
			'''
			########################################SALMAP DIST########################################################
			cls1 = "mediumslateblue"
			cls2 = "firebrick"
			
			plt.figure(figsize=(10, 10), dpi=80)
			plt.grid(True)
			plt.hist(dist_salmap_1_from_cls_1,ec="k",alpha=1,bins=50,color="mediumslateblue")
			plt.hist(dist_salmap_1_from_cls_2,ec="k",alpha=0.7,bins=50,color="firebrick")
			handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [cls1,cls2]]
			labels= ["Class 1","Class 2"]
			plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
			#plt.ylim(0,100)
			#plt.xlim(0,5)
			plt.xlabel('Euclidian distance',fontweight='bold',fontsize=12)
			plt.ylabel("Counts",fontweight='bold',fontsize=12)
			plt.title('Euclidian distance of Subpopulation 1 Saliency Map to Test Set', fontweight='bold',fontsize=15)
			plt.show()
				
			
			plt.figure(figsize=(10, 10), dpi=80)
			plt.grid(True)
			plt.hist(dist_salmap_2_from_cls_1,ec="k",alpha=1,bins=50,color="mediumslateblue")
			plt.hist(dist_salmap_2_from_cls_2,ec="k",alpha=0.7,bins=50,color="firebrick")
			handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [cls1,cls2]]
			labels= ["Class 1","Class 2"]
			plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
			#plt.ylim(0,100)
			#plt.xlim(0,5)
			plt.xlabel('Euclidian distance',fontweight='bold',fontsize=12)
			plt.ylabel("Counts",fontweight='bold',fontsize=12)
			plt.title('Euclidian distance of Subpopulation 2 Saliency Map to Test Set', fontweight='bold',fontsize=15)
			plt.show()
			'''
			########################################MEAN DIST########################################################
			'''
			cls1 = "mediumslateblue"
			cls2 = "firebrick"
			
			plt.figure(figsize=(10, 10), dpi=80)
			plt.grid(True)
			plt.hist(dist_mean_1_from_cls_1,ec="k",alpha=1,bins=50,color="mediumslateblue")
			plt.hist(dist_mean_1_from_cls_2,ec="k",alpha=0.7,bins=50,color="firebrick")
			handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [cls1,cls2]]
			labels= ["Class 1","Class 2"]
			plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
			#plt.ylim(0,100)
			#plt.xlim(0,5)
			plt.xlabel('Euclidian distance',fontweight='bold',fontsize=12)
			plt.ylabel("Counts",fontweight='bold',fontsize=12)
			plt.title('Euclidian distance of Class 1 Mean to Test Set', fontweight='bold',fontsize=15)
			plt.show()
				
			
			plt.figure(figsize=(10, 10), dpi=80)
			plt.grid(True)
			plt.hist(dist_mean_2_from_cls_1,ec="k",alpha=1,bins=50,color="mediumslateblue")
			plt.hist(dist_mean_2_from_cls_2,ec="k",alpha=0.7,bins=50,color="firebrick")
			handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [cls1,cls2]]
			labels= ["Class 1","Class 2"]
			plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
			#plt.ylim(0,100)
			#plt.xlim(0,5)
			plt.xlabel('Euclidian distance',fontweight='bold',fontsize=12)
			plt.ylabel("Counts",fontweight='bold',fontsize=12)
			plt.title('Euclidian distance of Class 2 Mean to Test Set', fontweight='bold',fontsize=15)
			plt.show()
			
		
		
			plt.imshow(testset_norm_cls_1_mean.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))),cmap=plt.cm.binary_r)
			plt.show()
			plt.imshow(testset_norm_cls_2_mean.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))),cmap=plt.cm.binary_r)
			plt.show()
			plt.imshow(testset_norm_mean.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))),cmap=plt.cm.binary_r)
			plt.show()
			
			'''
		
		
		mean_cumweight1=np.mean(np.array(cumweights1),axis=0)
		
		plt.figure(figsize = (12,10),dpi=100)
		plt.imshow(mean_cumweight1,cmap=plt.cm.binary_r)
		plt.title("Mean cumulative weight representation map for subpopulation 1, "+str(digit_1)+" VS "+str(digit_2),fontsize=14,fontweight="bold")
		plt.suptitle("No Turnover, Synapses:" +str(syn), y=0.93,fontsize=14,fontweight="bold")
		if turn==0:
			plt.suptitle("Turnover, Synapses:" +str(syn), y=0.93,fontsize=14,fontweight="bold")						
		#plt.savefig("./temp/mean_repmap_sub_1_digits_"+str(digit_1)+"_vs_"+str(digit_2)+ "_turnover_" + str(turn) +"_synapses_"+str(syn) +".png",bbox_inches='tight')
		plt.show()
		
		
		
		mean_cumweight2=np.mean(np.array(cumweights2),axis=0)

		plt.figure(figsize = (12,10),dpi=100)
		plt.imshow(mean_cumweight2,cmap=plt.cm.binary_r)
		plt.title("Mean cumulative weight representation map for subpopulation 2, "+str(digit_1)+" VS "+str(digit_2),fontsize=14,fontweight="bold")
		plt.suptitle("No Turnover, Synapses:" +str(syn), y=0.93,fontsize=14,fontweight="bold")
		if turn==0:
			plt.suptitle("Turnover, Synapses:" +str(syn), y=0.93,fontsize=14,fontweight="bold")		
		#plt.savefig("./temp/mean_repmap_sub_2_digits_"+str(digit_1)+"_vs_"+str(digit_2)+ "_turnover_" + str(turn) +"_synapses_"+str(syn) +".png",bbox_inches='tight')
		plt.show()
		

		
		
		turn_dists_mean_11.append(runs_dists_mean_11)				
		turn_dists_std_11.append(runs_dists_std_11)
		
		turn_dists_mean_12.append(runs_dists_mean_12)				
		turn_dists_std_12.append(runs_dists_std_12)
		
		turn_dists_mean_21.append(runs_dists_mean_21)				
		turn_dists_std_21.append(runs_dists_std_21)
		
		turn_dists_mean_22.append(runs_dists_mean_22)				
		turn_dists_std_22.append(runs_dists_std_22)

	mean_dist_sal_11_list.append(turn_dists_mean_11)
	mean_dist_sal_12_list.append(turn_dists_mean_12)
	mean_dist_sal_21_list.append(turn_dists_mean_21)
	mean_dist_sal_22_list.append(turn_dists_mean_22)
		
	std_dist_sal_11_list.append(turn_dists_std_11)
	std_dist_sal_12_list.append(turn_dists_std_12)
	std_dist_sal_21_list.append(turn_dists_std_21)
	std_dist_sal_22_list.append(turn_dists_std_22)
	


	

	

mean_dist_sal_11_meanarr = np.mean(np.array(mean_dist_sal_11_list),axis=2)
mean_dist_sal_12_meanarr = np.mean(np.array(mean_dist_sal_12_list),axis=2)
mean_dist_sal_21_meanarr = np.mean(np.array(mean_dist_sal_21_list),axis=2)
mean_dist_sal_22_meanarr = np.mean(np.array(mean_dist_sal_22_list),axis=2)




inclass_dist=(np.array(mean_dist_sal_11_list) + np.array(mean_dist_sal_22_list))/2
inclass_dist=inclass_dist/np.max(inclass_dist)


acrossclass_dist=(np.array(mean_dist_sal_12_list) + np.array(mean_dist_sal_21_list))/2
acrossclass_dist=acrossclass_dist/np.max(acrossclass_dist)


pairslab=[(str(pair[0]) + " VS " + str(pair[1])) for pair in pairs]

turnlab=["Turnover","No Turnover"]


dfpair=[]
dfturn=[]
dfincl=[]
dfacrcl=[]

for i in range(len(pairs)):
	dfpair+=40*[pairslab[i]]
	dfturn+=20*[turnlab[0]] + 20*[turnlab[1]]
	dfincl+=inclass_dist[0][i].tolist() + inclass_dist[1][i].tolist()
	dfacrcl+=acrossclass_dist[0][i].tolist() + acrossclass_dist[1][i].tolist()


df_plt={"Digits":dfpair,
		"Turn":dfturn,
		"Normalized Within Class Distance":dfincl,
		"Normalized Across Class Distance":dfacrcl
		}



df_plt=pd.DataFrame(df_plt)

df_plt.to_pickle('./temp/Representation_Map_Distance_Boxplots_Synapses_'+str(syn))


ax = sns.catplot(x='Digits', y='Normalized Within Class Distance', hue='Turn', 
            data=df_plt, kind='box', height=8,aspect = 2, legend=True,
            legend_out = False,dodge=True,width=0.5)
plt.suptitle('Within Class Distance of Representation Maps to Test Set, Synapses:'+ str(syn),fontsize=14,fontweight="bold")
plt.legend(loc=0, prop={'size': 10})
#plt.ylim(5,10)
#plt.savefig("./temp/Within_Class_Distance_Shapes_Synapses_"+ str(syn) +".png",bbox_inches='tight')
#plt.savefig("./temp/Within_Class_Distance_Shapes_Synapses_"+ str(syn) +".svg",bbox_inches='tight')
plt.show()



ax = sns.catplot(x='Digits', y='Normalized Across Class Distance', hue='Turn', 
            data=df_plt, kind='box', height=8,aspect = 2, legend=True,
            legend_out = False,dodge=True,width=0.5)
plt.suptitle('Across Class Distance of Representation Maps to Test Set, Synapses:'+ str(syn),fontsize=14,fontweight="bold")
plt.legend(loc=0, prop={'size': 10})
#plt.ylim(5,10)
#plt.savefig("./temp/Across_Class_Distance_Shapes_Synapses_"+ str(syn) +".png",bbox_inches='tight')
#plt.savefig("./temp/Across_Class_Distance_Shapes_Synapses_"+ str(syn) +".svg",bbox_inches='tight')
plt.show()























'''
turnticks=[1,2]
plt.figure(figsize=(10, 8), dpi=80)
for i in range(len(distdiff_sub1[0])):
	plt.plot(turnticks, distdiff_sub1[:,i], 'ko-',ms=10, alpha=1)
plt.boxplot(distdiff_sub1.T)
#plt.yticks(np.arange(50,110,10),fontsize=10,fontweight="bold")
plt.ylabel('Euclidean Distance Difference per Class', fontweight = 'bold',fontsize=12)
#plt.ylim(-1,110)
plt.xticks(turnticks,["Turnover","No\nTurnover"],fontweight="bold",fontsize=12)
plt.title("Mean Distance Differnce across Turnovers Subpopulation 1", fontsize=14, fontweight='bold')	
#plt.savefig("./temp/Mean_Dots_Shapes_Synapses_"+ str(s) +".png",bbox_inches='tight')
plt.show()


turnticks=[1,2]
plt.figure(figsize=(10, 8), dpi=80)
for i in range(len(distdiff_sub2[0])):
	plt.plot(turnticks, distdiff_sub2[:,i], 'ko-',ms=10, alpha=1)
plt.boxplot(distdiff_sub2.T)
#plt.yticks(np.arange(50,110,10),fontsize=10,fontweight="bold")
plt.ylabel('Euclidean Distance Difference per Class', fontweight = 'bold',fontsize=12)
#plt.ylim(-1,110)
plt.xticks(turnticks,["Turnover","No\nTurnover"],fontweight="bold",fontsize=12)
plt.title("Mean Distance Differnce across Turnovers Subpopulation 2", fontsize=14, fontweight='bold')	
#plt.savefig("./temp/Mean_Dots_Shapes_Synapses_"+ str(s) +".png",bbox_inches='tight')
plt.show()
'''
