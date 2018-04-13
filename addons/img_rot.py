import cv2
import os,sys
import numpy as np
import time
import imutils

src=sys.argv[1]
dst=sys.argv[2]
n=int(sys.argv[3])
dangle=360.0/n

if not os.path.isdir(dst):
    os.mkdir(dst)

for cls in os.listdir(src):
    cls_path=os.path.join(src,cls)
    dst_cls_path=os.path.join(dst,cls)
    if not os.path.isdir(dst_cls_path):
        os.mkdir(dst_cls_path)
    for fname in os.listdir(cls_path):
        file_path=os.path.join(cls_path,fname)
        img=cv2.imread(file_path)
        count=1
        millis=int(round(time.time()*1000))
        dst_file_path=os.path.join(dst_cls_path,'%ld_%d.jpg'%(millis,count))
        cv2.imwrite(dst_file_path,img)
        count+=1
        angle=dangle
        for i in range(int(np.floor(n))):
            dst_file_path=os.path.join(dst_cls_path,'%ld_%d.jpg'%(millis,count+i))
            r_img=imutils.rotate_bound(img,angle)
            angle+=dangle
            cv2.imwrite(dst_file_path,r_img)
print 'finished'