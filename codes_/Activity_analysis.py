import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns


data_file = "./results/test_activity_asdf.txt"

df=pd.read_table(data_file)


digit1='1'
digit2='2'

active_thresh=10
ndends=10
npyrs=80


dfd1=df[df['LABEL']==digit1]
spikesumd1=np.zeros(ndends*npyrs)
c=0
div=0

for i in range(ndends*npyrs,dfd1.shape[0],ndends*npyrs):
	div+=1
	temp_dfd1=dfd1.iloc()[c:i]
	spikesumd1+=temp_dfd1['NEURON_SPIKES'].values.astype(int)
	c=i

spikesumd1/=div

dfd2=df[df['LABEL']==digit2]
spikesumd2=np.zeros(ndends*npyrs)
c=0
div=0

for i in range(ndends*npyrs,dfd2.shape[0],ndends*npyrs):
	div+=1
	temp_dfd2=dfd2.iloc()[c:i]
	spikesumd2+=temp_dfd2['NEURON_SPIKES'].values.astype(int)
	c=i

spikesumd2/=div


neuronspikesd1=np.zeros(80)
neuronspikesd2=np.zeros(80)
j=0


for i in range(0,len(spikesumd1),ndends):
	neuronspikesd1[j]=round(spikesumd1[i])
	neuronspikesd2[j]=round(spikesumd2[i])
	j+=1




neuronspikesd1sub1=neuronspikesd1[:40].reshape(40,1)
neuronspikesd1sub2=neuronspikesd1[40:].reshape(40,1)


neuronspikesd2sub1=neuronspikesd2[:40].reshape(40,1)
neuronspikesd2sub2=neuronspikesd2[40:].reshape(40,1)


meanneuronspikesperdigit=np.vstack((neuronspikesd1,neuronspikesd2)).T


meanneuronspikespersubpopd1=np.hstack((neuronspikesd1sub1,neuronspikesd1sub2))
meanneuronspikespersubpopd2=np.hstack((neuronspikesd2sub1,neuronspikesd2sub2))

meanneuronspikespersubpopperdigit=np.hstack((meanneuronspikespersubpopd1,meanneuronspikespersubpopd2))

plt.figure(figsize=(20,16),dpi=80)
ax = sns.heatmap(meanneuronspikespersubpopperdigit, cmap="coolwarm", vmin=0 , vmax = (2*active_thresh)-1)

ax.set_xticklabels([1,2,1,2],fontsize=16,fontweight='bold')
ax.set_yticklabels(range(1,41), rotation = 0,fontsize=16,fontweight='bold')
ax.set_xlabel('SUBPOPULATION',fontsize=16,fontweight='bold')
ax.set_ylabel('NEURON',fontsize=16,fontweight='bold')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
plt.title('MEAN ACTIVITY PER CLASS PER SUBPOPULATION', fontweight= "bold",fontsize=20,loc='center')

ax.axvline(2, color='black', lw=5)
ax2=ax.twiny()
ax2.get_xaxis().set_ticks([1,3,4])
ax2.set_xlabel('PRESENTED CLASS',fontsize=16,fontweight='bold')
ax2.set_xticklabels([1,2,np.nan],fontsize=16,fontweight='bold')
#plt.show()
plt.savefig("./temp/Activity_1",bbox_inches='tight')



plt.figure(figsize=(20,16),dpi=80)
ax = sns.heatmap(meanneuronspikesperdigit, cmap="coolwarm", vmin=0 , vmax = (2*active_thresh)-1)

ax.set_xticklabels([1,2],fontsize=16,fontweight='bold')
ax.get_yaxis().set_ticks(np.arange(0.5,80.5,1))
ax.set_yticklabels(range(1,81), rotation = 0,fontsize=12,fontweight='bold')
ax.set_xlabel('PRESENTED CLASS',fontsize=16,fontweight='bold')
ax.set_ylabel('NEURON',fontsize=16,fontweight='bold')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
plt.title('MEAN ACTIVITY PER CLASS', fontweight= "bold",fontsize=20,loc='center')
ax.axvline(1, color='black', lw=5)

#plt.show()
plt.savefig("./temp/Activity_2",bbox_inches='tight')


'''
overlap_perc_tot = 100*np.sum(np.sum(meanneuronspikesperdigit>20,axis=1)==2)/len(meanneuronspikesperdigit)


overlap_perc_per_active_in_one_subpop = 100*np.sum(np.sum(meanneuronspikesperdigit>20,axis=1)==2)/np.sum(np.sum(meanneuronspikesperdigit>20,axis=1)>=1)




plt.figure(figsize=(5,50),dpi=180)
ax = sns.heatmap(meanneuronspikesperdigit, cmap="coolwarm", vmin=0 , vmax = 39)

ax.set_xticklabels([3,8],fontsize=12,fontweight='bold')
ax.set_xlabel('PRESENTED DIGIT',fontsize=12,fontweight='bold')
ax.axvline(1, color='black', lw=5)

plt.show()
'''

'''
##SNIPPET FOR TOTAL MEAN ACTIVITY
spikesum=np.zeros(800)
for i in range(800,5000,801):
	div+=1
	if c==0:
		temp_df=df.iloc()[c:i]
	else:
		temp_df=df.iloc()[c+1:i]		
	#print(df.iloc()[c:i])
	spikesum+=temp_df['NEURON_SPIKES'].values.astype(np.int)
	#print(spikesum/div)
	c=i

	
spikesum/=div

neuronspikes=np.zeros(80)

j=0
for i in range(0,len(spikesum),10):
	neuronspikes[j]=spikesum[i]
	j+=1
'''
	


fig, axs = plt.subplots(2,2, figsize=(16, 16), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0, wspace=0)
#fig.tight_layout()
plt.setp(axs, xticks=range(0,41,2), xlim=(-1,41),ylim=(0,45),yticks=range(0,45,5))
fig.suptitle("Mean Activity Histograms",fontsize=30,fontweight='bold')
fig.text(0.5, 0.085, 'Mean Activity', ha='center',fontsize=20,fontweight='bold')

axs = axs.ravel()

axs[0].set_xticks(())
axs[1].set_xticks(())
axs[1].set_yticks(())
axs[3].set_yticks(())

axs[0].set_title("Subpopulation 1",fontsize=20,fontweight='bold')
axs[0].set_ylabel("Class 1",fontsize=20,fontweight='bold')
axs[1].set_title("Subpopulation 2",fontsize=20,fontweight='bold')
axs[2].set_ylabel("Class 2",fontsize=20,fontweight='bold')


for i in range(len(meanneuronspikespersubpopperdigit.T)):
	axs[i].hist(meanneuronspikespersubpopperdigit[:,i],bins=(np.arange(-1,41)-0.5),color='blue',edgecolor='black')
	axs[i].plot((active_thresh-0.5,active_thresh-0.5),(45,0),color='r',linestyle='--')

plt.savefig("./temp/Activity_hist",bbox_inches='tight')