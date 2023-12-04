import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns


# Inputs
data_file_w = "./results/presyn_w_sub1_asdf.txt"

data_file_w2 = "./results/presyn_w_sub2_asdf.txt"

# Delimiter
data_file_delimiter = '\t'

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
df_w_tot = pd.read_csv(data_file_w, header=None, delimiter=data_file_delimiter, names=column_names)
df_w_tot=df_w_tot.fillna(0)




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
df_w2_tot = pd.read_csv(data_file_w2, header=None, delimiter=data_file_delimiter, names=column_names)
df_w2_tot=df_w2_tot.fillna(0)





#My_dfs
inputs=400
iters=20

for part in range(0,len(df_w_tot),inputs):
	df_w=df_w_tot[part:part+inputs]
	
	#df_nrn
	#df_br
	
	df_w2=df_w2_tot[part:part+inputs]
	#df_nrn2
	#df_br2
	
	input_ids=df_w[0]
	
	w_tab_1=df_w.values[:,1:]
	w_tab_2=df_w2.values[:,1:]
	
	
	
	#w_tab_1[w_tab_1<0.3]=0
	#w_tab_2[w_tab_2<0.3]=0
	
	
	cumweight1=np.sum(w_tab_1,axis=1).reshape((int(np.sqrt(inputs)),int(np.sqrt(inputs))))
	cumweight2=np.sum(w_tab_2,axis=1).reshape((int(np.sqrt(inputs)),int(np.sqrt(inputs))))
	
	maxweight1=np.max(w_tab_1,axis=1).reshape((int(np.sqrt(inputs)),int(np.sqrt(inputs))))
	maxweight2=np.max(w_tab_2,axis=1).reshape((int(np.sqrt(inputs)),int(np.sqrt(inputs))))
	
	
	numsyns1=np.sum(w_tab_1!=0,axis=1).reshape((int(np.sqrt(inputs)),int(np.sqrt(inputs))))
	numsyns2=np.sum(w_tab_2!=0,axis=1).reshape((int(np.sqrt(inputs)),int(np.sqrt(inputs))))
	
	
	cumweighttot=cumweight1+cumweight2
	maxweighttot=maxweight1+maxweight2
	numsynstot=numsyns1+numsyns2
	
	
	plt.figure(figsize = (12,10),dpi=100)
	plt.imshow(cumweight1,cmap=plt.cm.binary_r)
	#plt.plot(range(19),range(19,0,-1),c='red',linewidth=2,linestyle="--")
	plt.title("Cumulative weight saliency map subpopulation 1 after "+str(iters)+" images",fontsize=14,fontweight="bold")
	#plt.savefig("./temp/Salience_cumw_sub1_"+str(iters),dpi=100,bbox_inches='tight')
	plt.show()
	
	
	plt.figure(figsize = (12,10),dpi=100)
	plt.imshow(cumweight2,cmap=plt.cm.binary_r)
	#plt.plot(range(19),range(19,0,-1),c='red',linewidth=2,linestyle="--")
	plt.title("Cumulative weight saliency map subpopulation 2 after "+str(iters)+" images",fontsize=14,fontweight="bold")
	#plt.savefig("./temp/Salience_cumw_sub2_"+str(iters),dpi=100,bbox_inches='tight')
	plt.show()
	
	
	plt.figure(figsize = (12,10),dpi=100)
	plt.imshow(cumweighttot,cmap=plt.cm.binary_r)
	#plt.plot(range(19),range(19,0,-1),c='red',linewidth=2,linestyle="--")
	plt.title("Cumulative weight saliency map total after "+str(iters)+" images",fontsize=14,fontweight="bold")
	#plt.savefig("./temp/Salience_cumw_tot_"+str(iters),dpi=100,bbox_inches='tight')
	plt.show()
	
	'''
	plt.figure(figsize = (12,10))
	plt.imshow(maxweight1,cmap=plt.cm.binary_r)
	plt.title("Cumulative weight saliency map subpopulation 1",fontsize=15,fontweight="bold")
	#plt.savefig("./temp/Salience_cumw_sub1_"+str(iters),bbox_inches='tight')
	plt.show()


	plt.figure(figsize = (12,10))
	plt.imshow(maxweight2,cmap=plt.cm.binary_r)
	plt.title("Cumulative weight saliency map subpopulation 2",fontsize=15,fontweight="bold")
	#plt.savefig("./temp/Salience_cumw_sub2_"+str(iters),bbox_inches='tight')
	plt.show()
	
	
	plt.figure(figsize = (12,10))
	plt.imshow(maxweighttot,cmap=plt.cm.binary_r)
	plt.title("Cumulative weight saliency map total",fontsize=15,fontweight="bold")
	#plt.savefig("./temp/Salience_cumw_tot_"+str(iters),bbox_inches='tight')
	plt.show()
	

	plt.figure(figsize = (12,10))
	plt.imshow(numsyns1,cmap=plt.cm.binary_r)
	plt.title("Number of synapses saliency map subpopulation 1",fontsize=15,fontweight="bold")
	#plt.savefig("./temp/Salience_nsyns_sub1_"+str(iters),bbox_inches='tight')
	plt.show()
	
	plt.figure(figsize = (12,10))
	plt.imshow(numsyns2,cmap=plt.cm.binary_r)
	plt.title("Number of synapses saliency map subpopulation 2",fontsize=15,fontweight="bold")
	#plt.savefig("./temp/Salience_nsyns_sub2_"+str(iters),bbox_inches='tight')
	plt.show()

	plt.figure(figsize = (12,10))
	plt.imshow(numsynstot,cmap=plt.cm.binary_r)
	plt.title("Number of synapses saliency map total",fontsize=15,fontweight="bold")
	#plt.savefig("./temp/Salience_nsyns_tot_"+str(iters),bbox_inches='tight')
	plt.show()
	'''
	iters+=20
	
'''	
np.save("./SalMapsNpy/Cumulative_weight_SalMap_Sub_1", cumweight1)
np.save("./SalMapsNpy/Cumulative_weight_SalMap_Sub_2", cumweight2)
np.save("./SalMapsNpy/Cumulative_weight_SalMap_total", cumweighttot)
'''