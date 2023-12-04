import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
import numpy as np

####MNIST####
#syns=[875,1750,3500]
#syns=[3500]
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
	#df_plt=pd.read_pickle('./temp/MNIST/Full/Accuracy_Boxplots/Accuracy_Boxplots_Synapses_'+str(syn))
	#asp=5
	#ymin=41
	
	##6_pairs 
	#df_plt=pd.read_pickle('./temp/MNIST/6_pairs/Accuracy_Boxplots/Accuracy_Boxplots_Synapses_'+str(syn))
	#asp=2
	#ymin=41
	
		
	###Constrained(ONLY 3500 SYNAPSES)
	
	##Full 
	#df_plt=pd.read_pickle('./temp/MNIST/3500_Constrained/Full/Accuracy_Boxplots/Accuracy_Boxplots_Synapses_'+str(syn))
	#asp=5
	#ymin=41
	
	##6_pairs
	#df_plt=pd.read_pickle('./temp/MNIST/3500_Constrained/6_pairs/Accuracy_Boxplots/Accuracy_Boxplots_Synapses_'+str(syn))
	#asp=2
	#ymin=41
	
	
	##Linear_Net(ONLY 1750 SYNAPSES)
	
	##Full 
	#df_plt=pd.read_pickle('./temp/MNIST/Linear_Net/Full/Accuracy_Boxplots/Accuracy_Boxplots_Synapses_'+str(syn))
	#asp=5	
	#ymin=15	
			
	##6_pairs 
	#df_plt=pd.read_pickle('./temp/MNIST/Linear_Net/6_pairs/Accuracy_Boxplots/Accuracy_Boxplots_Synapses_'+str(syn))
	#asp=2
	#ymin=15	
	
	
	#############
	
	####SHAPES####	
	df_plt=pd.read_pickle('./temp/Shapes/3_diffs/Accuracy_Boxplots/Accuracy_Boxplots_Synapses_'+str(syn))
	asp=2
	##############
	
	ax = sns.catplot(x=xlab, y="% Accuracy", hue='Turn', 
	            data=df_plt, kind='box', height=8,aspect = asp, legend=True,
	            legend_out = False,dodge=True,width=0.5)
	plt.suptitle('Accuracy per Digit Pair across 20 runs, Synapses: ' + str(syn),fontsize=14,fontweight="bold")
	plt.legend(loc=0, prop={'size': 10})
	#plt.ylim(ymin,105) #COMMENT OUT FOR SHAPES
	plt.savefig("./temp/Accuracy_Boxplots_Synapses_"+ str(syn) +".png",bbox_inches='tight')
	plt.savefig("./temp/Accuracy_Boxplots_Synapses_"+ str(syn) +".svg",bbox_inches='tight')
	plt.show()
