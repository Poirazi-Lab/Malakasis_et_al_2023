import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
from scipy.spatial import distance
from matplotlib.patches import Rectangle
from mnist import MNIST



class_1=1
class_2=2

#FOR SHAPES
'''
data_file = "./Shapes_train"
df=pd.read_csv(data_file,header=None)
trainset=df.values


data_file = "./Shapes_train_labels"
df=pd.read_csv(data_file,header=None)
trainset_labels=df.values.reshape(df.values.shape[1],)
'''


data_file = "./Shape_Datasets/Shapes_test_041"
df=pd.read_csv(data_file,header=None)
testset=df.values


data_file = "./Shape_Datasets/Shapes_test_labels_041"
df=pd.read_csv(data_file,header=None)
testset_labels=df.values.reshape(df.values.shape[1],)

dim=400



'''
#FOR MNIST
mndata=MNIST('./MNIST/')
imagelists,labels=mndata.load_testing()
labels=np.array(labels)
cls1_indexes = np.where(labels==class_1)[0]
cls2_indexes = np.where(labels==class_2)[0]
x=np.hstack((cls1_indexes,cls2_indexes))
x.sort()
testset=np.array(imagelists)[x]
testset_labels=labels[x]


dim=784
'''

testset_norm=testset/np.max(testset)


testset_norm_cls_1=testset_norm[testset_labels==class_1]
testset_norm_cls_2=testset_norm[testset_labels==class_2]

testset_norm_mean=np.mean(testset_norm,axis=0)
testset_norm_cls_1_mean=np.mean(testset_norm_cls_1,axis=0)
testset_norm_cls_2_mean=np.mean(testset_norm_cls_2,axis=0)

plt.imshow(testset_norm_cls_1_mean.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))),cmap=plt.cm.binary_r)
plt.show()
plt.imshow(testset_norm_cls_2_mean.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))),cmap=plt.cm.binary_r)
plt.show()
plt.imshow(testset_norm_mean.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))),cmap=plt.cm.binary_r)
plt.show()

cumweight1 = np.load("./SalMapsNpy/Cumulative_weight_SalMap_Sub_1.npy")
cumweight2 = np.load("./SalMapsNpy/Cumulative_weight_SalMap_Sub_2.npy")
cumweighttot = np.load("./SalMapsNpy/Cumulative_weight_SalMap_total.npy")

cumweight1_norm = cumweight1/np.max(cumweight1)
cumweight2_norm = cumweight2/np.max(cumweight2)
cumweighttot_norm = cumweighttot/np.max(cumweighttot)


cumweight1_norm_flat = cumweight1_norm.flatten()
cumweight2_norm_flat = cumweight2_norm.flatten()
cumweighttot_norm_flat = cumweighttot_norm.flatten()



dist_salmap_1=[]
dist_salmap_2=[]
dist_salmap_tot=[]

dist_mean_1=[]
dist_mean_2=[]
dist_mean_tot=[]

for i in testset_norm:
	dist_salmap_1.append(distance.euclidean(cumweight1_norm_flat, i))
	dist_salmap_2.append(distance.euclidean(cumweight2_norm_flat, i))
	dist_salmap_tot.append(distance.euclidean(cumweighttot_norm_flat, i))

	dist_mean_1.append(distance.euclidean(testset_norm_cls_1_mean, i))
	dist_mean_2.append(distance.euclidean(testset_norm_cls_2_mean, i))
	dist_mean_tot.append(distance.euclidean(testset_norm_mean, i))
	
	
dist_salmap_1=np.array(dist_salmap_1)
dist_salmap_2=np.array(dist_salmap_2)
dist_salmap_tot=np.array(dist_salmap_tot)


dist_mean_1=np.array(dist_mean_1)
dist_mean_2=np.array(dist_mean_2)
dist_mean_tot=np.array(dist_mean_tot)



plt.imshow(cumweight1_norm_flat.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))),cmap=plt.cm.binary_r)
plt.show()
plt.imshow(cumweight2_norm_flat.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))),cmap=plt.cm.binary_r)
plt.show()
plt.imshow(cumweighttot_norm_flat.reshape(int(np.sqrt(dim)),int(np.sqrt(dim))),cmap=plt.cm.binary_r)
plt.show()



dist_salmap_1_from_cls_1 = dist_salmap_1[testset_labels==class_1]
dist_salmap_1_from_cls_2 = dist_salmap_1[testset_labels==class_2]
dist_salmap_2_from_cls_1 = dist_salmap_2[testset_labels==class_1]
dist_salmap_2_from_cls_2 = dist_salmap_2[testset_labels==class_2]

dist_mean_1_from_cls_1 = dist_mean_1[testset_labels==class_1]
dist_mean_1_from_cls_2 = dist_mean_1[testset_labels==class_2]
dist_mean_2_from_cls_1 = dist_mean_2[testset_labels==class_1]
dist_mean_2_from_cls_2 = dist_mean_2[testset_labels==class_2]




mean_dist_sal_11=np.mean(dist_salmap_1_from_cls_1)
std_dist_sal_11=np.std(dist_salmap_1_from_cls_1)

mean_dist_sal_12=np.mean(dist_salmap_1_from_cls_2)
std_dist_sal_12=np.std(dist_salmap_1_from_cls_2)

mean_dist_sal_21=np.mean(dist_salmap_2_from_cls_1)
std_dist_sal_21=np.std(dist_salmap_2_from_cls_1)

mean_dist_sal_22=np.mean(dist_salmap_2_from_cls_2)
std_dist_sal_22=np.std(dist_salmap_2_from_cls_2)




mean_dist_mean_11=np.mean(dist_mean_1_from_cls_1)
std_dist_mean_11=np.std(dist_mean_1_from_cls_1)

mean_dist_mean_12=np.mean(dist_mean_1_from_cls_2)
std_dist_mean_12=np.std(dist_mean_1_from_cls_2)

mean_dist_mean_21=np.mean(dist_mean_2_from_cls_1)
std_dist_mean_21=np.std(dist_mean_2_from_cls_1)

mean_dist_mean_22=np.mean(dist_mean_2_from_cls_2)
std_dist_mean_22=np.std(dist_mean_2_from_cls_2)



print("Representations:")
print("Mean distance of representation 1 to class 1: ",mean_dist_sal_11," ± ", std_dist_sal_11)
print("Mean distance of representation 1 to class 2: ",mean_dist_sal_12," ± ", std_dist_sal_12)
print("\n")
print("Mean distance of representation 2 to class 1: ",mean_dist_sal_21," ± ", std_dist_sal_21)
print("Mean distance of representation 2 to class 2: ",mean_dist_sal_22," ± ", std_dist_sal_22)
print("\n")
print("Means:")
print("Mean distance of mean 1 to class 1: ",mean_dist_mean_11," ± ", std_dist_mean_11)
print("Mean distance of mean 1 to class 2: ",mean_dist_mean_12," ± ", std_dist_mean_12)
print("\n")
print("Mean distance of mean 2 to class 1: ",mean_dist_mean_21," ± ", std_dist_mean_21)
print("Mean distance of mean 2 to class 2: ",mean_dist_mean_22," ± ", std_dist_mean_22)



########################################SALMAP DIST########################################################
cls1 = "mediumslateblue"
cls2 = "firebrick"

plt.figure(figsize=(10, 10), dpi=80)
plt.grid(True)
plt.hist(dist_salmap_1_from_cls_1,ec="k",alpha=1,bins=50,color="mediumslateblue")
plt.hist(dist_salmap_1_from_cls_2,ec="k",alpha=0.7,bins=50,color="firebrick")
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [cls1,cls2]]
labels= ["Class 1","Class 2"]
plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
#plt.ylim(0,100)
#plt.xlim(0,5)
plt.xlabel('Euclidian distance',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.title('Euclidian distance of Subpopulation 1 Saliency Map to Test Set', fontweight='bold',fontsize=15)
plt.show()
	

plt.figure(figsize=(10, 10), dpi=80)
plt.grid(True)
plt.hist(dist_salmap_2_from_cls_1,ec="k",alpha=1,bins=50,color="mediumslateblue")
plt.hist(dist_salmap_2_from_cls_2,ec="k",alpha=0.7,bins=50,color="firebrick")
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [cls1,cls2]]
labels= ["Class 1","Class 2"]
plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
#plt.ylim(0,100)
#plt.xlim(0,5)
plt.xlabel('Euclidian distance',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.title('Euclidian distance of Subpopulation 2 Saliency Map to Test Set', fontweight='bold',fontsize=15)
plt.show()


plt.figure(figsize=(10, 10), dpi=80)
plt.grid(True)
plt.hist(dist_salmap_tot,ec="k",alpha=1,bins=50,color="darkorchid")
#plt.hist(dist_salmap_tot[testset_labels==class_1],ec="k",alpha=1,bins=50,color="mediumslateblue")
#plt.hist(dist_salmap_tot[testset_labels==class_2],ec="k",alpha=0.7,bins=50,color="firebrick")
plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
#plt.ylim(0,100)
#plt.xlim(0,5)
plt.xlabel('Euclidian distance',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.title('Euclidian distance of Total Saliency Map to Test Set', fontweight='bold',fontsize=15)
plt.show()


########################################MEAN DIST########################################################

cls1 = "mediumslateblue"
cls2 = "firebrick"

plt.figure(figsize=(10, 10), dpi=80)
plt.grid(True)
plt.hist(dist_mean_1_from_cls_1,ec="k",alpha=1,bins=50,color="mediumslateblue")
plt.hist(dist_mean_1_from_cls_2,ec="k",alpha=0.7,bins=50,color="firebrick")
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [cls1,cls2]]
labels= ["Class 1","Class 2"]
plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
#plt.ylim(0,100)
#plt.xlim(0,5)
plt.xlabel('Euclidian distance',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.title('Euclidian distance of Class 1 Mean to Test Set', fontweight='bold',fontsize=15)
plt.show()
	

plt.figure(figsize=(10, 10), dpi=80)
plt.grid(True)
plt.hist(dist_mean_2_from_cls_1,ec="k",alpha=1,bins=50,color="mediumslateblue")
plt.hist(dist_mean_2_from_cls_2,ec="k",alpha=0.7,bins=50,color="firebrick")
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [cls1,cls2]]
labels= ["Class 1","Class 2"]
plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
#plt.ylim(0,100)
#plt.xlim(0,5)
plt.xlabel('Euclidian distance',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.title('Euclidian distance of Class 2 Mean Saliency Map to Test Set', fontweight='bold',fontsize=15)
plt.show()


plt.figure(figsize=(10, 10), dpi=80)
plt.grid(True)
plt.hist(dist_mean_tot,ec="k",alpha=1,bins=50,color="darkorchid")
#plt.hist(dist_mean_tot[testset_labels==class_1],ec="k",alpha=1,bins=50,color="mediumslateblue")
#plt.hist(dist_mean_tot[testset_labels==class_2],ec="k",alpha=0.7,bins=50,color="firebrick")
plt.legend(handles, labels,fontsize=12,prop={'weight':'bold','size':12})
#plt.ylim(0,100)
#plt.xlim(0,5)
plt.xlabel('Euclidian distance',fontweight='bold',fontsize=12)
plt.ylabel("Counts",fontweight='bold',fontsize=12)
plt.title('Euclidian distance of Total Input to Test Set', fontweight='bold',fontsize=15)
plt.show()

#########################################################################################################