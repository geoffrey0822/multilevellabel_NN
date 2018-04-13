import os,sys
import numpy as np

gest_sample={}
label_at=int(sys.argv[2])
with open(sys.argv[1],'r') as f:
    for ln in f:
        line=ln.rstrip('\n')
        labels=line.split(',')[1]
        label=int(labels.split(';')[label_at])
        if label in gest_sample.keys():
            gest_sample[label]+=1
        else:
            gest_sample[label]=1
print gest_sample