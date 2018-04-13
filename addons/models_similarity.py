import os,sys
import numpy as np
from scipy import signal

def createPairs(n):
    pairs=np.zeros((n*n,2),dtype=int)
    c=0
    for i in range(n):
        for j in range(n):
            pairs[c,0]=i
            pairs[c,1]=j
            c+=1
    return pairs

nModel=len(sys.argv)-1
feats=[]
model_names=[]
nData=0
for i in range(nModel):
    name,ext=os.path.splitext(sys.argv[i+1])
    model_feat=np.loadtxt(sys.argv[i+1],delimiter=',')
    feats.append(model_feat)
    print name
    print model_feat.shape
    #print '%s:%d'%(name,model_feat.shape[0]*model_feat.shape[1])
    model_names.append(name)
    if i==0:
        nData=model_feat.shape[0]
nRow=len(model_names)
pairs=createPairs(nRow)
print pairs
corrs=np.zeros((nRow,nRow),dtype='float')
rr=0
cc=0
for r in range(pairs.shape[0]):
    stdA=feats[pairs[r,0]].std()
    stdB=feats[pairs[r,1]].std()
    corr=signal.correlate2d(feats[pairs[r,0]]/stdA,feats[pairs[r,1]]/stdB,mode='valid')
    #corr=np.corrcoef(feats[pairs[r,0]]/stdA,feats[pairs[r,1]]/stdB)
    #print corr.shape
    corrs[rr,cc]=corr
    cc+=1
    if cc!=0 and cc%nRow==0:
        cc=0
        rr+=1
    print 'correlation:%s<--->%s\t\t=\t%f'%(model_names[pairs[r,0]],model_names[pairs[r,1]],corr)
print corrs
print corr.shape  
print 'finish'