import os,sys
import numpy as np
import re
import shutil
from scanf import scanf

src=sys.argv[1]
dst=sys.argv[2]
mode=sys.argv[3]
split=15
cls_fmt=''
at=0
if mode=='num':
    if len(sys.argv)>4:
        split=sys.argv[4]
elif mode=='cls':
    if len(sys.argv)>4:
        cls_fmt=sys.argv[4]
    if len(sys.argv)>5:
        at=int(sys.argv[5])

if not os.path.isdir(dst):
    os.mkdir(dst)

count=0
if mode=='num':
    cls_id=0
    current_dir=os.path.join(dst,str(cls_id))
for fname in sorted(os.listdir(src),key=lambda x:(int(re.sub('\D','',x)),x)):
    if mode=='num':
        if count%split==0:
            cls_id+=1
            current_dir=os.path.join(dst,str(cls_id))
            os.mkdir(current_dir)
        shutil.copy(os.path.join(src,fname),current_dir)
        count+=1
    else:
        cls_id=scanf(cls_fmt,fname)[at]
        #print scanf(cls_fmt,fname)
        current_dir=os.path.join(dst,str(cls_id))
        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)
        shutil.copy(os.path.join(src,fname),current_dir)
    
print 'finished'