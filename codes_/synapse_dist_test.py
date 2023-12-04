import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections


df=pd.read_table("./target_neuron_ids_asdf.txt",header=None)
df=df.drop(df.columns[[len(df.values[0])-1]],axis=1)

df2=pd.read_table("./weights_per_epoch_asdf.txt",header=None)
df2=df2.drop(df2.columns[[len(df2.values[0])-1]],axis=1)


x=df.values[0]
y=df2.values[1]


df3=pd.DataFrame(np.vstack((x,y)).T,columns=['nid','weight'])



collections.Counter(df3[df3['weight']<=0.3]['nid'].values)



keys=list(collections.Counter(x).keys())
vals=list(collections.Counter(x).values())



'''
plt.figure(figsize=(15, 30), dpi=80)
plt.scatter(vals,keys,s=5,c="red")
plt.grid(True)
plt.ylabel('Neuron ID')
plt.yticks(range(0,200,1))
plt.xticks(range(8,29,1))
#plt.title('Weight Distribution')
plt.show()
'''


	