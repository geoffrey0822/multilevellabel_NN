import os,sys
import numpy as np
from scipy import signal
from pandas import DataFrame
import sympy

def createPairs(n):
    pairs=np.zeros((n*n,2),dtype=int)
    c=0
    for i in range(n):
        for j in range(n):
            pairs[c,0]=i
            pairs[c,1]=j
            c+=1
    return pairs


coeffs=np.loadtxt(sys.argv[1],delimiter=',')
title=sys.argv[2]
accuracy=float(sys.argv[3])
seg_coeffs=np.vsplit(coeffs,coeffs.shape[0]/coeffs.shape[1])
print coeffs.shape
print len(seg_coeffs)
avg_coeffs=np.zeros((coeffs.shape[1],coeffs.shape[1]),dtype='float')
for coeff in seg_coeffs:
    avg_coeffs=np.add(avg_coeffs,coeff)
avg_coeffs/=len(seg_coeffs)

lbls=sympy.symbols('branch_1:%d'%(coeffs.shape[1]+1))
data_map=[]
for i in range(len(lbls)):
    data_map.append((lbls[i],avg_coeffs[i,:]))

coeffs_tab=DataFrame.from_items(data_map,columns= lbls, orient='index')
#print avg_coeffs
print '----------------------------------------------'
print 'Model Name:\t%s'%title
print 'Model Accuracy:\t%f'%accuracy
print '----------------------------------------------'
print coeffs_tab
print '----------------------------------------------'