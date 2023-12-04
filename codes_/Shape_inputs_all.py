import numpy as np
import csv
from skimage import draw


runs=range(1,21)
stabs=[0,1,2]
overlaps=[0,2,4]

rad=3
bright=False

for st in stabs:
	for ov in overlaps:
		for run in runs:


	
			runseed=int('2020'+str(run))
			np.random.seed(runseed)
	
			stab = st
			overlap = ov + stab
			
			cls_1=[]
			cls_2=[]
			
			
			for i in range(1200):
				img=np.zeros((20,20))
				img2=np.zeros((20,20))
				
			
				strt_x_1=np.random.randint(0+overlap+stab,10-2*rad+1+overlap-stab)
				strt_y_1=np.random.randint(1+overlap+stab,11-2*rad+1+overlap-stab)
				
				start = (strt_x_1,strt_y_1)
				extent = (3,6)
				rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)
				
				
				if bright:
					img[rr, cc] = np.random.randint(250,255,size=(img[rr, cc].shape))
				else:
					img[rr, cc] = np.random.randint(0,255,size=(img[rr, cc].shape))
				
				
				rr, cc = draw.rectangle_perimeter(start, extent=extent, shape=img.shape)
				
				
				img[rr, cc] = 255
				
				
				strt_x_2=np.random.randint(10+2*rad-1-overlap+stab,20-overlap-stab)
				strt_y_2=np.random.randint(10+2*rad-1-overlap+stab,20-overlap-stab)
				
				rr, cc = draw.disk((strt_x_2-rad,strt_y_2-rad), radius=rad, shape=img2.shape)
				
			
				if bright:
					img2[rr, cc] = np.random.randint(250,255,size=(img2[rr, cc].shape[0]))
				else:
					img2[rr, cc] = np.random.randint(0,255,size=(img2[rr, cc].shape[0]))		
			
			
				rr, cc = draw.circle_perimeter(strt_x_2-rad, strt_y_2-rad, rad,shape=img2.shape)
				img2[rr, cc] = 255
				
				
				
				cls_1.append(img.flatten())
				cls_2.append(img2.flatten())
				
			
				
			
			cls_1=np.array(cls_1).astype(int)
			cls_2=np.array(cls_2).astype(int)
			
			
			
			cls_1_tr=cls_1[200:]
			cls_1_te=cls_1[:200]
			
			cls_2_tr=cls_2[200:]
			cls_2_te=cls_2[:200]
			
			
			y1=np.ones(1000).astype(int)
			y2=2*np.ones(1000).astype(int)
			
			Xa=[]
			ya=[]
			
			for i in range(1000):
				Xa.append(cls_1_tr[i])
				Xa.append(cls_2_tr[i])
				ya.append(y1[i])
				ya.append(y2[i])
				
				
			X=np.array(Xa)
			y=ya
			
			
			
			
			
			
			with open('./Shape_Datasets/Shapes_train_labels_'+str(st)+str(ov)+str(run), 'w+') as f:
			    # create the csv writer
			    writer = csv.writer(f)
			
			    # write a row to the csv file
			    writer.writerow(y)
			
			with open('./Shape_Datasets/Shapes_train_'+str(st)+str(ov)+str(run), 'w+') as ff:
			    # create the csv writer
				writer = csv.writer(ff)
			
			    # write a row to the csv file
				for i in range(len(X)):
					writer.writerow(X[i])
			
			
					
			Xt=np.vstack((cls_1_te,cls_2_te))
			
			y1t=np.ones(200).astype(int)
			y2t=2*np.ones(200).astype(int)
			yt=np.hstack((y1t,y2t))
			
			idx_tr = np.random.permutation(len(Xt))
			
			Xt=Xt[idx_tr]
			yt=yt[idx_tr]
			
			
			
			
			
			
			
			with open('./Shape_Datasets/Shapes_test_labels_'+str(st)+str(ov)+str(run), 'w+') as f:
			    # create the csv writer
			    writer = csv.writer(f)
			
			    # write a row to the csv file
			    writer.writerow(yt)
			
			with open('./Shape_Datasets/Shapes_test_'+str(st)+str(ov)+str(run), 'w+') as ff:
			    # create the csv writer
				writer = csv.writer(ff)
			
			    # write a row to the csv file
				for i in range(len(Xt)):
					writer.writerow(Xt[i])
			
	
