#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
from matplotlib.patches import Rectangle

####MNIST####
#syns=[875,1750,3500]
#syns=[3500]
#xlab="Digits"
#############
	
	
####SHAPES####
syns=[500,1000,2000]
xlab="Difficulty"
##############

	
for syn in syns:
	
	####MNIST####

	##Full
	#df_plt=pd.read_pickle('./temp/MNIST/Full/Effective_Weights/effective_weights_synapses_'+str(syn))
	#asp=5
	
	##6_pairs
	#df_plt=pd.read_pickle('./temp/MNIST/6_pairs/Effective_Weights/effective_weights_synapses_'+str(syn))
	#asp=2
	
	###Constrained
	
	##6_pairs
	#df_plt=pd.read_pickle('./temp/MNIST/3500_Constrained/6_pairs/Effective_Weights/effective_weights_synapses_'+str(syn))
	#asp=2
	#############
	
	
	####SHAPES####
	df_plt=pd.read_pickle('./temp/Shapes/3_diffs/Effective_Weights/effective_weights_synapses_'+str(syn))
	asp=2
	##############
	
	
	
	
	ax = sns.catplot(x=xlab, y='Effective Weight %', hue='Turn', 
	            data=df_plt, kind='box', height=8,aspect = asp, legend=True,
	            legend_out = False,dodge=True,width=0.5)
	plt.suptitle('Effective Weights, Synapses:' + str(syn))
	plt.legend(loc=0, prop={'size': 10})
	plt.savefig("./temp/effective_weights_"+"_synapses_" + str(syn) + ".png",bbox_inches='tight')
	plt.savefig("./temp/effective_weights_"+"_synapses_" + str(syn) + ".svg",bbox_inches='tight')
	plt.show()
