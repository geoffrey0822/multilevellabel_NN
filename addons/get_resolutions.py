import os,sys
import numpy as np
import cv2

resolutions=[]

file_path=sys.argv[1]
rootpath=sys.argv[2]
output=sys.argv[3]

with open(output,'wb') as f:
    with open(file_path,'r') as rf:
        for ln in rf:
            line=ln.rstrip('\n')
            fields=line.split(',')
            img_path=os.path.join(rootpath,fields[0].replace('\\','/'))
            img=cv2.imread(img_path)
            res=img.shape[0:2]
            if res not in resolutions:
                f.write('%d,%d\n'%(res[0],res[1]))
                resolutions.append(res)
                print '%d,%d'%(res[0],res[1])
print 'finished'