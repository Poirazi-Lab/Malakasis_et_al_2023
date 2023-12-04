import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
from matplotlib.patches import Rectangle


data_file = "./results/test_activity_asdf.txt"

df=pd.read_table(data_file)

digit1='1'
digit2='2'



dfd1=df[df['LABEL']==digit1]
dfd2=df[df['LABEL']==digit2]




###############################################################################
c=0
corr_list_d1=[]
wrong_list_d1=[]

for i in range(800,dfd1.shape[0],800):

	temp_dfd1=dfd1.iloc()[c:i]
	
	spiked = temp_dfd1[temp_dfd1["BRANCH_SPIKES"]!='0']
	
	correct=spiked[spiked["SUBPOP"]==spiked["LABEL"]]
	wrong=spiked[spiked["SUBPOP"]!=spiked["LABEL"]]
	
	corr_list_d1.append(correct["BRANCHID"].values.astype(int))
	wrong_list_d1.append(wrong["BRANCHID"].values.astype(int))
	
	c=i


uniq_wrong_list_d1=[]
for i in wrong_list_d1:
	for j in i:
		if j not in uniq_wrong_list_d1:
			uniq_wrong_list_d1.append(j)


uniq_wrong_list_d1_str=[str(i) for i in sorted(uniq_wrong_list_d1)]



uniq_corr_list_d1=[]
for i in corr_list_d1:
	for j in i:
		if j not in uniq_corr_list_d1:
			uniq_corr_list_d1.append(j)


uniq_corr_list_d1_str=[str(i) for i in sorted(uniq_corr_list_d1)]




corr_bspikes_list_d1=[]
wrong_bspikes_list_d1=[]

corr_nspikes_list_d1=[]
wrong_nspikes_list_d1=[]

c=0
for i in range(800,dfd1.shape[0],800):

	temp_dfd1=dfd1.iloc()[c:i]
	
	correct=temp_dfd1[temp_dfd1["BRANCHID"].isin(uniq_corr_list_d1_str)]
	wrong=temp_dfd1[temp_dfd1["BRANCHID"].isin(uniq_wrong_list_d1_str)]
	corr_bspikes_list_d1.append((correct["BRANCHID"].values.astype(int),correct["BRANCH_SPIKES"].values.astype(int)))
	wrong_bspikes_list_d1.append((wrong["BRANCHID"].values.astype(int),wrong["BRANCH_SPIKES"].values.astype(int)))


	corr_nspikes_list_d1.append((correct["NEURONID"].values.astype(int),correct["NEURON_SPIKES"].values.astype(int)))
	wrong_nspikes_list_d1.append((wrong["NEURONID"].values.astype(int),wrong["NEURON_SPIKES"].values.astype(int)))	
	
	
	c=i




wrong_nspikes_arr_d1=np.array([np.unique(np.array(i),axis=1) for i in wrong_nspikes_list_d1])
corr_nspikes_arr_d1=np.array([np.unique(np.array(i),axis=1) for i in corr_nspikes_list_d1])

mean_corr_neural_activity_d1=np.mean(corr_nspikes_arr_d1,axis=0)
mean_wrong_neural_activity_d1=np.mean(wrong_nspikes_arr_d1,axis=0)

corr_bspikes_arr_d1=np.array([np.array(i) for i in corr_bspikes_list_d1])
wrong_bspikes_arr_d1=np.array([np.array(i) for i in wrong_bspikes_list_d1])

mean_corr_dend_activity_d1=np.mean(corr_bspikes_arr_d1,axis=0)
mean_wrong_dend_activity_d1=np.mean(wrong_bspikes_arr_d1,axis=0)

###############################################################################










###############################################################################
c=0
corr_list_d2=[]
wrong_list_d2=[]

for i in range(800,dfd2.shape[0],800):

	temp_dfd2=dfd2.iloc()[c:i]
	
	spiked = temp_dfd2[temp_dfd2["BRANCH_SPIKES"]!='0']
	
	correct=spiked[spiked["SUBPOP"]==spiked["LABEL"]]
	wrong=spiked[spiked["SUBPOP"]!=spiked["LABEL"]]
	
	corr_list_d2.append(correct["BRANCHID"].values.astype(int))
	wrong_list_d2.append(wrong["BRANCHID"].values.astype(int))
	
	c=i


uniq_wrong_list_d2=[]
for i in wrong_list_d2:
	for j in i:
		if j not in uniq_wrong_list_d2:
			uniq_wrong_list_d2.append(j)


uniq_wrong_list_d2_str=[str(i) for i in sorted(uniq_wrong_list_d2)]



uniq_corr_list_d2=[]
for i in corr_list_d2:
	for j in i:
		if j not in uniq_corr_list_d2:
			uniq_corr_list_d2.append(j)


uniq_corr_list_d2_str=[str(i) for i in sorted(uniq_corr_list_d2)]







corr_bspikes_list_d2=[]
wrong_bspikes_list_d2=[]

corr_nspikes_list_d2=[]
wrong_nspikes_list_d2=[]

c=0
for i in range(800,dfd2.shape[0],800):

	temp_dfd2=dfd2.iloc()[c:i]
	
	correct=temp_dfd2[temp_dfd2["BRANCHID"].isin(uniq_corr_list_d2_str)]
	wrong=temp_dfd2[temp_dfd2["BRANCHID"].isin(uniq_wrong_list_d2_str)]
	corr_bspikes_list_d2.append((correct["BRANCHID"].values.astype(int),correct["BRANCH_SPIKES"].values.astype(int)))
	wrong_bspikes_list_d2.append((wrong["BRANCHID"].values.astype(int),wrong["BRANCH_SPIKES"].values.astype(int)))



	corr_nspikes_list_d2.append((correct["NEURONID"].values.astype(int),correct["NEURON_SPIKES"].values.astype(int)))
	wrong_nspikes_list_d2.append((wrong["NEURONID"].values.astype(int),wrong["NEURON_SPIKES"].values.astype(int)))	
	
	
	c=i


wrong_nspikes_arr_d2=np.array([np.unique(np.array(i),axis=1) for i in wrong_nspikes_list_d2])
corr_nspikes_arr_d2=np.array([np.unique(np.array(i),axis=1) for i in corr_nspikes_list_d2])

mean_corr_neural_activity_d2=np.mean(corr_nspikes_arr_d2,axis=0)
mean_wrong_neural_activity_d2=np.mean(wrong_nspikes_arr_d2,axis=0)

corr_bspikes_arr_d2=np.array([np.array(i) for i in corr_bspikes_list_d2])
wrong_bspikes_arr_d2=np.array([np.array(i) for i in wrong_bspikes_list_d2])

mean_corr_dend_activity_d2=np.mean(corr_bspikes_arr_d2,axis=0)
mean_wrong_dend_activity_d2=np.mean(wrong_bspikes_arr_d2,axis=0)


###############################################################################



###############################################################################
flat_wrong_d1=np.array([i for j in wrong_list_d1 for i in j])
flat_corr_d1=np.array([i for j in corr_list_d1 for i in j])
flat_wrong_d2=np.array([i for j in wrong_list_d2 for i in j])
flat_corr_d2=np.array([i for j in corr_list_d2 for i in j])




neurons_wd1=set(np.floor(flat_wrong_d1/10))
dends_wd1=np.array(sorted(list(set(flat_wrong_d1))))
neurons_wd1_dups=np.floor(np.array(sorted(list(dends_wd1)))/10)


neurons_cd1=set(np.floor(flat_corr_d1/10))
dends_cd1=np.array(sorted(list(set(flat_corr_d1))))
neurons_cd1_dups=np.floor(np.array(sorted(list(dends_cd1)))/10)



neurons_wd2=set(np.floor(flat_wrong_d2/10))
dends_wd2=np.array(sorted(list(set(flat_wrong_d2))))
neurons_wd2_dups=np.floor(np.array(sorted(list(dends_wd2)))/10)



neurons_cd2=set(np.floor(flat_corr_d2/10))
dends_cd2=np.array(sorted(list(set(flat_corr_d2))))
neurons_cd2_dups=np.floor(np.array(sorted(list(dends_cd2)))/10)


d1 = "mediumslateblue"
d2 = "firebrick"
s1 = "aqua"
s2 = "lightcoral"

plt.figure(figsize=(20, 20), dpi=80)
plt.scatter(dends_cd1,neurons_cd1_dups,ec="b",alpha=1, c=d1,s=100,marker="D")
plt.scatter(dends_wd2,neurons_wd2_dups,ec="r",alpha=1,c=d2,s=50,marker="d")
labels= ["Class 1(correct)","Class 2(wrong)"]
handles = [Rectangle((0,0),1,1,color=c,ec=k) for c,k in [(d1,"b"),(d2,"r")]]
plt.legend(handles, labels,fontsize=22,prop={'weight':'bold','size':25},loc=2)
plt.ylim(-2,42)
plt.xlim(-20,420)
plt.tick_params(labelsize=22)
plt.xlabel('Dendrite Index',fontweight='bold',fontsize=22)
plt.ylabel("Neuron Index",fontweight='bold',fontsize=22)
plt.title('Active Dendrites in Subpopulation 1',fontweight='bold',fontsize=25)
plt.savefig("./temp/Sub1Dends.svg",bbox_inches='tight')
plt.savefig("./temp/Sub1Dends.png",bbox_inches='tight')
plt.show()

plt.figure(figsize=(20, 20), dpi=80)
plt.scatter(dends_cd2,neurons_cd2_dups,ec="r",alpha=1,c=d2,s=100,marker="D")
plt.scatter(dends_wd1,neurons_wd1_dups,ec="b",alpha=1, c=d1,s=50,marker="d")
handles = [Rectangle((0,0),1,1,color=c,ec=k) for c,k in [(d1,"b"),(d2,"r")]]
labels= ["Class 1(wrong)","Class 2(correct)"]
plt.legend(handles, labels,fontsize=22,prop={'weight':'bold','size':25},loc=2)
plt.ylim(38,82)
plt.xlim(380,820)
plt.tick_params(labelsize=22)
plt.xlabel('Dendrite Index',fontweight='bold',fontsize=22)
plt.ylabel("Neuron Index",fontweight='bold',fontsize=22)
plt.title('Active Dendrites in Subpopulation 2',fontweight='bold',fontsize=25)
plt.savefig("./temp/Sub2Dends.svg",bbox_inches='tight')
plt.savefig("./temp/Sub2Dends.png",bbox_inches='tight')
plt.show()

###############################################################################

mean_corr_dend_activity_d1
mean_wrong_dend_activity_d1

mean_corr_dend_activity_d2
mean_wrong_dend_activity_d2


plt.figure(figsize=(20, 20), dpi=80)

plt.scatter(mean_corr_neural_activity_d1[0]*10 +5,mean_corr_neural_activity_d1[1],ec="b",alpha=1, c=s1,s=700,marker="s")
plt.scatter(mean_wrong_neural_activity_d2[0]*10 +5,mean_wrong_neural_activity_d2[1],ec="r",alpha=1, c=s2,s=700,marker="s")


plt.scatter(mean_corr_dend_activity_d1[0],mean_corr_dend_activity_d1[1],ec="b",alpha=1, c=d1,s=100,marker="d")
plt.scatter(mean_wrong_dend_activity_d2[0],mean_wrong_dend_activity_d2[1],ec="r",alpha=1,c=d2,s=50,marker="d")

handles = [Rectangle((0,0),1,1,color=c,ec=k) for c,k in [(d1,"b"),(d2,"r"),(s1,"b"),(s2,"r")]]
labels= ["Class 1 dend","Class 2 dend","Class 1 soma","Class 2 soma"]


plt.legend(handles, labels,fontsize=22,prop={'weight':'bold','size':25},loc=2)

plt.xticks(range(0,401,10),labels=range(1,42))
plt.yticks(range(0,30))
plt.grid(True)

plt.tick_params(labelsize=18)
plt.xlabel("Neuron Index",fontweight='bold',fontsize=22)
plt.ylabel("Mean Activity(Spikes/dSpikes)",fontweight='bold',fontsize=22)
plt.title('Mean Activity in Subpopulation 1',fontweight='bold',fontsize=25)

plt.savefig("./temp/Sub1Activity.svg",bbox_inches='tight')
plt.savefig("./temp/Sub1Activity.png",bbox_inches='tight')
plt.show()


plt.figure(figsize=(20, 20), dpi=80)


plt.scatter(mean_corr_neural_activity_d2[0]*10 +5,mean_corr_neural_activity_d2[1],ec="r",alpha=1, c=s2, s=650,marker="s")
plt.scatter(mean_wrong_neural_activity_d1[0]*10 +5,mean_wrong_neural_activity_d1[1],ec="b",alpha=1, c=s1, s=650,marker="s")


plt.scatter(mean_corr_dend_activity_d2[0],mean_corr_dend_activity_d2[1],ec="r",alpha=1,c=d2,s=100,marker="d") 
plt.scatter(mean_wrong_dend_activity_d1[0],mean_wrong_dend_activity_d1[1],ec="b",alpha=1, c=d1,s=50,marker="d")
plt.xticks(range(400,801,10),labels=range(41,82))
plt.yticks(range(0,30))
plt.grid(True)


handles = [Rectangle((0,0),1,1,color=c,ec=k) for c,k in [(d1,"b"),(d2,"r"),(s1,"b"),(s2,"r")]]
labels= ["Class 1 dend","Class 2 dend","Class 1 soma","Class 2 soma"]


plt.legend(handles, labels,fontsize=22,prop={'weight':'bold','size':25},loc=2)

plt.tick_params(labelsize=18)
plt.xlabel("Neuron Index",fontweight='bold',fontsize=22)
plt.ylabel("Mean Activity(Spikes/dSpikes)",fontweight='bold',fontsize=22)
plt.title('Mean Activity in Subpopulation 2',fontweight='bold',fontsize=25)

plt.savefig("./temp/Sub2Activity.svg",bbox_inches='tight')
plt.savefig("./temp/Sub2Activity.png",bbox_inches='tight')
plt.show()



mean_corr_neural_activity_d1
mean_wrong_neural_activity_d1
mean_corr_neural_activity_d2
mean_wrong_neural_activity_d2
###############################################################################
############################PREPOST ANALYSIS###################################

data_file_prepost = "./results/prepost_w_asdf.txt"
df_prepost=pd.read_table(data_file_prepost,header=None)
df_prepost.columns=["pre","post","w"]

df_prepost_eff=df_prepost[df_prepost["w"]>0.3]

df_prepost_eff_sub1=df_prepost_eff[df_prepost_eff['post']<400]
df_prepost_eff_sub2=df_prepost_eff[df_prepost_eff['post']>=400]


pre_wrong_d1=list(set(df_prepost_eff[df_prepost_eff["post"].isin(mean_wrong_dend_activity_d1[0])]["pre"].values))
pre_corr_d1=list(set(df_prepost_eff[df_prepost_eff["post"].isin(mean_corr_dend_activity_d1[0])]["pre"].values))
pre_wrong_d2=list(set(df_prepost_eff[df_prepost_eff["post"].isin(mean_wrong_dend_activity_d2[0])]["pre"].values))
pre_corr_d2=list(set(df_prepost_eff[df_prepost_eff["post"].isin(mean_corr_dend_activity_d2[0])]["pre"].values))


data_file = "./Shapes_test"
df=pd.read_csv(data_file,header=None)
testset=df.values
dim=400

data_file = "./Shapes_test_labels"
df=pd.read_csv(data_file,header=None)
testset_labels=df.values.reshape(dim,)


#DO PER SUBPOP
prepost_cumw=[]
for i in range(dim):
	prepost_cumw.append(np.sum(df_prepost_eff[df_prepost_eff["pre"]==i]['w']))



prepost_cumw_sub1=[]
for i in range(dim):
	prepost_cumw_sub1.append(np.sum(df_prepost_eff_sub1[df_prepost_eff_sub1["pre"]==i]['w']))


prepost_cumw_sub2=[]
for i in range(dim):
	prepost_cumw_sub2.append(np.sum(df_prepost_eff_sub2[df_prepost_eff_sub2["pre"]==i]['w']))




test_mean=np.mean(testset,axis=0)

plt.figure(figsize=(20, 20), dpi=80)
ax= sns.heatmap(test_mean.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))), linewidth=10, cmap="binary_r",cbar=False)

for i in pre_corr_d1:
	if i not in pre_wrong_d2:
		y=i//np.sqrt(dim)
		x=i%np.sqrt(dim)
		ax.add_patch(Rectangle((x,y), 1, 1, fill=False, edgecolor='blue',alpha=0.5, lw=prepost_cumw_sub1[i]))


for i in pre_wrong_d2:
	y=i//np.sqrt(dim)
	x=i%np.sqrt(dim)
	ax.add_patch(Rectangle((x,y), 1, 1, fill=False, edgecolor='red',alpha=0.5, lw=prepost_cumw_sub1[i]))
	
plt.title("Cumulative weights to Subpopulation 1",fontweight='bold',fontsize=25)
plt.savefig("./temp/Sub1_cumw.svg",bbox_inches='tight')
plt.savefig("./temp/Sub1_cumw.png",bbox_inches='tight')
plt.show()


plt.figure(figsize=(20, 20), dpi=80)
ax= sns.heatmap(test_mean.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))), linewidth=10, cmap="binary_r",cbar=False)


for i in pre_corr_d2:
	if i not in pre_wrong_d1:
		y=i//np.sqrt(dim)
		x=i%np.sqrt(dim)
		ax.add_patch(Rectangle((x,y), 1, 1, fill=False, edgecolor='red', alpha=0.5,lw=prepost_cumw_sub2[i]))


for i in pre_wrong_d1:
	y=i//np.sqrt(dim)
	x=i%np.sqrt(dim)
	ax.add_patch(Rectangle((x,y), 1, 1, fill=False, edgecolor='blue',alpha=0.5, lw=prepost_cumw_sub2[i]))

		
plt.title("Cumulative weights to Subpopulation 2",fontweight='bold',fontsize=25)
plt.savefig("./temp/Sub2_cumw.svg",bbox_inches='tight')
plt.savefig("./temp/Sub2_cumw.png",bbox_inches='tight')
plt.show()



###############################################################################

'''



for i in range(800,dfd1.shape[0],800):

	temp_dfd1=dfd1.iloc()[c:i]
	
	spiked = temp_dfd1[temp_dfd1["BRANCH_SPIKES"]!='0']
	
	correct=spiked[spiked["SUBPOP"]==spiked["LABEL"]]
	wrong=spiked[spiked["SUBPOP"]!=spiked["LABEL"]]

	
	corr_bspikes_list_d1.append(correct["BRANCH_SPIKES"].values.astype(int))
	wrong_bspikes_list_d1.append(wrong["BRANCH_SPIKES"].values.astype(int))


	corr_nspikes_list_d1.append((correct["NEURONID"].values.astype(int),correct["NEURON_SPIKES"].values.astype(int)))
	wrong_nspikes_list_d1.append((wrong["NEURONID"].values.astype(int),wrong["NEURON_SPIKES"].values.astype(int)))	
	
	corr_list_d1.append(correct["BRANCHID"].values.astype(int))
	wrong_list_d1.append(wrong["BRANCHID"].values.astype(int))
	
	
	
	c=i


for i in range(800,dfd2.shape[0],800):

	temp_dfd2=dfd2.iloc()[c:i]
	
	spiked = temp_dfd2[temp_dfd2["BRANCH_SPIKES"]!='0']
	
	correct=spiked[spiked["SUBPOP"]==spiked["LABEL"]]
	wrong=spiked[spiked["SUBPOP"]!=spiked["LABEL"]]


	corr_bspikes_list_d2.append(correct["BRANCH_SPIKES"].values.astype(int))
	wrong_bspikes_list_d2.append(wrong["BRANCH_SPIKES"].values.astype(int))


	corr_nspikes_list_d2.append((correct["NEURONID"].values.astype(int),correct["NEURON_SPIKES"].values.astype(int)))
	wrong_nspikes_list_d2.append((wrong["NEURONID"].values.astype(int),wrong["NEURON_SPIKES"].values.astype(int)))	
	
	
	corr_list_d2.append(correct["BRANCHID"].values.astype(int))
	wrong_list_d2.append(wrong["BRANCHID"].values.astype(int))
	
	c=i
'''




