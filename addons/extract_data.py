import os,sys
import shutil
import numpy as np

src=sys.argv[1]
dst=sys.argv[2]
root_path=sys.argv[3]

if not os.path.isdir(dst):
    os.mkdir(dst)
with open(src,'r') as f:
    print src
    count=0
    for ln in f:
        line=ln.rstrip('\n')
        fields=line.split(',')
        img_path=os.path.join(root_path,fields[0])
        lbl=fields[1].split(';')[1]
        out_path=os.path.join(dst,lbl)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        shutil.copy(img_path, out_path)
        count+=1
        if count%100==0:
            print 'exported %d data'%count
        
print 'finish'