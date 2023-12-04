import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
DF LABELS:
"INPUT_IDX", "INPUT_FR", "HIDDEN_IDX", "HIDDEN_FR",  "HIDDEN_ACTIVE", 
"HIDDEN_SUBPOP" , "WEIGHT", "ITERATION", "LABEL"
'''

train=pd.read_table("./synaptic_table_train_asdf.txt",header=0)
train=train.drop(train.columns[[len(train.values[0])-1]],axis=1)

test=pd.read_table("./synaptic_table_test_asdf.txt",header=0)
test=test.drop(test.columns[[len(test.values[0])-1]],axis=1)


train[train["HIDDEN_ACTIVE"]!=0][train["INPUT_FR"]!=0][train["HIDDEN_SUBPOP"]==0][train["ITERATION"]==2]["HIDDEN_FR"]


train[train["HIDDEN_ACTIVE"]!=0][train["INPUT_FR"]==0][train["HIDDEN_SUBPOP"]==0][train["ITERATION"]==1]["HIDDEN_FR"]


