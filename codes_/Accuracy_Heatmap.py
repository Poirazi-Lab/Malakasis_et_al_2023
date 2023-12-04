import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


meantable=np.load("meantable.npy")
stdtable=np.load("stdtable.npy")

plt.figure(figsize=(10, 8), dpi=80)
ax = sns.heatmap(meantable, linewidth=0.5, cmap="inferno",annot=stdtable, vmin=50 , vmax = 100)
ax.yaxis.tick_right()
ax.xaxis.tick_top()


ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=12,fontweight='bold')
ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9], rotation = 0,fontsize=12,fontweight='bold')
plt.title('Mean Accuracy Heatmap', fontweight= "bold",fontsize=14,loc='center')

plt.suptitle('Mean Accuracy Heatmap', fontweight= "bold", x= 0.435, fontsize=18)
plt.title('(std in boxes)', fontsize=13,fontweight='bold')

plt.show()


