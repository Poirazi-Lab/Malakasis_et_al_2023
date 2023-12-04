import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
from scipy import stats


full_path_data_1 ="/media/nikos/Elements/ClusterResults/results_261121_AllDigits_20runs_8280interstim_constrained_turnover_alternating_strongLTD/"
full_path_data_2 ="/media/nikos/Elements/ClusterResults/results_261121_AllDigits_20runs_8280interstim_constrained_turnover_alternating_maxCREB/"
file_type ="predictons"
suffix = "_nikos_"
bad_seeds=[]
acc_tables_list=[]
std_tables_list=[]
all_accs_list=[]
pvals=[]

for full_path_data in [full_path_data_1,full_path_data_2]:
	accs_list=[]
	for d2 in range(10):
		for d1 in range(d2):	
			
			digit1=str(d1)
			digit2=str(d2)
			#plt.figure(figsize=(10, 10), dpi=80) ##FOR INDIVIDUAL PLOTS PER DIGIT PAIR
			
			accs=[]
			for s in range(1,21):
				
				if s in bad_seeds:
					continue
				seed = str(s)
				df=pd.read_table(full_path_data + file_type + suffix + seed + "_digits_" 
			      + digit1 + "_" + digit2 + ".txt",header=None, error_bad_lines=False, sep = " ")	
	
	
	
	
	
			
				
				#THIS IS FOR FINAL ACCURACY ANALYSIS
				#REQUIRES FILE TYPE = predictons!
				acc=round(sum(df[2])/len(df[2]), 2)
				accs.append(acc)
			accs_list.append(accs)
	
	acc_table=np.array(accs_list)
	
	all_accs_list.append(acc_table)

		
		
	mean_accs=np.mean(acc_table,axis = 1)
	accs_std=np.std(acc_table,axis = 1)
	
	
	meantable = np.zeros((10,10))
	stdtable = 	np.zeros((10,10))
	
	c=0
	
	for d2 in range(10):
		for d1 in range(d2):	
			meantable[d1,d2] = 100*mean_accs[c]
			stdtable[d1,d2] = 100*accs_std[c]
			c += 1
			
	acc_tables_list.append(meantable)
	std_tables_list.append(stdtable)
	

	
	
mean_acc_diff=acc_tables_list[0]-acc_tables_list[1]
std_diff=std_tables_list[0]-std_tables_list[1]




for i in range(len(all_accs_list[0])):
	pval=stats.ttest_ind(all_accs_list[0][i],all_accs_list[1][i],equal_var=False)[1]
	pvals.append(pval)


pvalstab = np.zeros((10,10))
pvalstars = np.full([10, 10], "", dtype=np.object)

c=0
for d2 in range(10):
	for d1 in range(d2):	
		pvalstab[d1,d2] = pvals[c]
		if 0.01<=pvals[c]<0.05:
			pvalstars[d1,d2] = "*"
		elif 0.001<=pvals[c]<0.01:
			pvalstars[d1,d2] = "**"
		elif 0.0001<=pvals[c]<0.001:
			pvalstars[d1,d2] = "***"
		elif pvals[c]<0.0001:
			pvalstars[d1,d2] = "****"
		elif pvals[c]>=0.05:
			pvalstars[d1,d2] = "ns"
		c+=1






plt.figure(figsize=(10, 8), dpi=80)
ax = sns.heatmap(mean_acc_diff, linewidth=0.5, cmap="coolwarm", annot=pvalstars, fmt='', vmin=-np.max(np.abs(mean_acc_diff)) , vmax = np.max(np.abs(mean_acc_diff)))
ax.yaxis.tick_right()
ax.xaxis.tick_top()


ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=12,fontweight='bold')
ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9], rotation = 0,fontsize=12,fontweight='bold')

plt.suptitle('Mean Accuracy Difference Heatmap', fontweight= "bold", x= 0.435, fontsize=18)
plt.title('(Control - Overexpressed CREB)', fontsize=13,fontweight='bold')

plt.show()




plt.figure(figsize=(10, 8), dpi=80)
ax = sns.heatmap(std_diff, linewidth=0.5, cmap="coolwarm",annot=pvalstars, fmt='', vmin=-np.max(np.abs(std_diff)) , vmax = np.max(np.abs(std_diff)))
ax.yaxis.tick_right()
ax.xaxis.tick_top()


ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=12,fontweight='bold')
ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9], rotation = 0,fontsize=12,fontweight='bold')

plt.suptitle('Standard Deviation Difference Heatmap', fontweight= "bold", x= 0.435, fontsize=18)
plt.title('(Control - Overexpressed CREB)', fontsize=13,fontweight='bold')

plt.show()


##TRYING OUT MERGED PLOT

mergedtab=np.zeros((10,10))

for d2 in range(10):
	for d1 in range(d2):	
		mergedtab[d1,d2]=mean_acc_diff[d1,d2]
		mergedtab[d2,d1]=std_diff[d1,d2]
		

plt.figure(figsize=(10, 8), dpi=80)
ax = sns.heatmap(mergedtab, linewidth=0.5, cmap="coolwarm",annot=pvalstars, fmt='', vmin=-np.max(np.abs(mean_acc_diff)) , vmax = np.max(np.abs(mean_acc_diff)))
ax.yaxis.tick_right()
ax.xaxis.tick_top()


ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=12,fontweight='bold')
ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9], rotation = 0,fontsize=12,fontweight='bold')



plt.suptitle('Accuracy and Standard Deviation Difference Heatmap', fontweight= "bold", x= 0.435, fontsize=16)
plt.title('(Control - Overexpressed CREB)', fontsize=13,fontweight='bold')

plt.show()







'''
std_diffs_list=[]
for i in std_diff:
	for j in i:
		if j!=0:
			std_diffs_list.append(j)
	

plt.figure(figsize=(10, 8), dpi=80)
plt.boxplot(std_diffs_list, notch=0, sym='+', vert=1, whis=1.5)
'''