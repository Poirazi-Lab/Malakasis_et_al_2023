import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_table("./results/test_spike_table_asdf.txt",header=None, error_bad_lines=False)

c=100

for i in range(81*0,81*10,81):
	fig, ax = plt.subplots(figsize=(20, 10))
	plt.title("Spike distribution, iteration " + str(c))
	plt.xlabel("spikes")
	plt.ylabel("neurons")
	plt.yticks(range(40))
	plt.xticks(range(0,130,10),rotation=50)
	tempdf = df[i:i+81]	
	new_header = tempdf.iloc[0] 
	tempdf = tempdf[1:] 
	tempdf.columns = new_header 
	tempdf=tempdf.astype('int')
	#tempdf[tempdf["SUBPOP"]==0].groupby("SPIKES").count().plot(kind='bar')
	for j in [0,5]:
		tempdf[tempdf["SUBPOP"]==j]["SPIKES"].plot.hist(alpha=0.8,rwidth=10,bins=100,label=str(j))
	ax.legend(loc='upper right')
	ax.grid(axis="y")
	#plt.savefig("./Data_From_Previous_Runs/RecordRunData(InfoOnSublime)/SpikeDistSample/SpikeDist_" + str(c))
	c+=1
