import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns



df=pd.read_table("./results/predictons_asdf.txt",header=None,sep=" ")

class_1=0
class_2=1




corr_class_1=sum(df[df[2]==1][1]==class_1)
corr_class_2=sum(df[df[2]==1][1]==class_2)

err_class_1=sum(df[df[2]==0][1]==class_2)
err_class_2=sum(df[df[2]==0][1]==class_1)


acc=(corr_class_1+corr_class_2)/len(df[0])

conf_matrix=np.array([[corr_class_1,err_class_2],[err_class_1,corr_class_2]])


df_cm = pd.DataFrame(conf_matrix, index = [3,8],
                  columns = [3,8])
plt.figure(figsize = (10,8))

sns.heatmap(df_cm, annot=True, fmt='.3g',cmap='viridis')
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Actual", fontsize=12, fontweight='bold')
plt.ylabel("Predicted", fontsize=12, fontweight='bold')
plt.xticks([0.5,1.5],[1,2],fontsize=12,fontweight='bold')
plt.yticks([0.5,1.5],[1,2],fontsize=12,fontweight='bold')
plt.show()
#plt.savefig("./temp/Confusion",bbox_inches='tight')


Mean_Conf = np.mean(df[3])
Std_Conf = np.std(df[3])


plt.figure(figsize = (10,8))
plt.violinplot(df[3])
plt.title("Confidence", fontsize=14, fontweight='bold')
plt.show()

print("Mean Confidence = ",round(Mean_Conf,2),"±",round(Std_Conf,2))
print("Accuracy=",round(100*acc,2),"%")


df=pd.read_table("./results/sample_accuracies_asdf.txt",header=None,sep=" ")


sample_interval=20


plt.figure(figsize=(10, 10), dpi=80)

y=np.vstack(([[0]],df.values))
x = list(range(sample_interval*0,sample_interval*len(y),sample_interval))				
plt.plot(x,y,c='k')
#plt.grid(True)
plt.xlabel('Training Iterations', fontweight = 'bold')
plt.yticks(np.arange(0,110,10))
plt.ylabel('% Accuracy', fontweight = 'bold')
plt.title('Learning Curve Linear Network', fontsize=14, fontweight='bold')
plt.show()


plt.close("all")


		

