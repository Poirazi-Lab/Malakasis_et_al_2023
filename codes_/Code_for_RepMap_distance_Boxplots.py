import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns


####MNIST####
#syns=[875,1750,3500]
#syns=[1750]
#xlab="Digits"
#############
	
	
####SHAPES####
syns=[500,1000,2000]
xlab="Difficulty"
##############


for syn in syns:
	
	####MNIST####
	##Full
	#df_plt=pd.read_pickle('./temp/MNIST/Full/Representation_Maps/Representation_Map_Distance_Boxplots_Synapses_'+str(syn))
	
	##6_pairs
	#df_plt=pd.read_pickle('./temp/MNIST/6_pairs/Representation_Maps/Representation_Map_Distance_Boxplots_Synapses_'+str(syn))


	###Linear Net(ONLY 1750 SYNAPSES)
	#6_pairs
	#df_plt=pd.read_pickle('./temp/MNIST/Linear_Net/6_pairs/Representation_Maps/Representation_Map_Distance_Boxplots_Synapses_'+str(syn))
	#############
	
	####SHAPES####
	df_plt=pd.read_pickle('./temp/Shapes/3_diffs/Representation_Maps/Representation_Map_Distance_Boxplots_Synapses_'+str(syn))
	##############
		
	ax = sns.catplot(x=xlab, y='Normalized Within Class Distance', hue='Turn', 
	            data=df_plt, kind='box', height=8,aspect = 2, legend=True,
	            legend_out = False,dodge=True,width=0.5)
	plt.suptitle('Within Class Distance of Representation Maps to Test Set, Synapses:'+ str(syn),fontsize=14,fontweight="bold")
	plt.legend(loc=0, prop={'size': 10})
	plt.savefig("./temp/Within_Class_Distance_Shapes_Synapses_"+ str(syn) +".png",bbox_inches='tight')
	plt.savefig("./temp/Within_Class_Distance_Shapes_Synapses_"+ str(syn) +".svg",bbox_inches='tight')
	plt.show()
	
	
	
	ax = sns.catplot(x=xlab, y='Normalized Across Class Distance', hue='Turn', 
	            data=df_plt, kind='box', height=8,aspect = 2, legend=True,
	            legend_out = False,dodge=True,width=0.5)
	plt.suptitle('Across Class Distance of Representation Maps to Test Set, Synapses:'+ str(syn),fontsize=14,fontweight="bold")
	plt.legend(loc=0, prop={'size': 10})
	plt.savefig("./temp/Across_Class_Distance_Shapes_Synapses_"+ str(syn) +".png",bbox_inches='tight')
	plt.savefig("./temp/Across_Class_Distance_Shapes_Synapses_"+ str(syn) +".svg",bbox_inches='tight')
	plt.show()
