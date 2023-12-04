import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#SPARSITY EXPERIMENT
perf_turn=[100,96.25,98.25,100,86.75]
perf_noturn=[99.75,97.5,93.75,49.75,0]
sparsity=[0,25,50,75,87.5]

plt.figure(figsize=(8,8),dpi=80)
plt.ylim((-1.5,105))
plt.ylabel("Accuracy %",fontsize=15,fontweight='bold')
plt.xlabel("Sparsity %",fontsize=15,fontweight='bold')
plt.title("Performance to Sparsity",fontsize=20,fontweight='bold')
plt.xlim((-1.5,105))
plt.plot(sparsity,perf_turn,marker="D",linestyle="--",c="black",label="Turnover")
plt.plot(sparsity,perf_noturn,marker="D",linestyle="--",c="r",label="No Turnover")
plt.legend(fontsize=15)
plt.show()


#STEP EXPERIMENT

#ACC TO BMI
perf=[100,93.5,86.75,56.75]
diff=[0.123,0.246,0.614,1.225]
step=[1,0.5,0.2,0.1]

plt.figure(figsize=(8,8),dpi=80)
plt.ylim((45,105))
plt.ylabel("Accuracy %",fontsize=15,fontweight='bold')
plt.xlabel("Difficulty (DBI)",fontsize=15,fontweight='bold')
plt.title("Performance to Difficulty",fontsize=20,fontweight='bold')
plt.xlim((0,2))
plt.plot(diff,perf,marker="D",linestyle="--",c="black")
plt.show()

#STEP TO BMI
plt.figure(figsize=(8,8),dpi=80)
plt.ylim((0,1.05))
plt.ylabel("Step",fontsize=15,fontweight='bold')
plt.xlabel("Difficulty (DBI)",fontsize=15,fontweight='bold')
plt.title("Step to Difficulty",fontsize=20,fontweight='bold')
plt.xlim((0,2))
plt.plot(diff,step,marker="D",linestyle="--",c="r")
plt.show()
