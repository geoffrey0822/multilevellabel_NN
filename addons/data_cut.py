import os,sys
import numpy as np
import shutil

src=sys.argv[1]
dst_1=sys.argv[2]
dst_2=sys.argv[3]
ratio=float(sys.argv[4])

if not os.path.isdir(dst_1):
    os.mkdir(dst_1)
if not os.path.isdir(dst_2):
    os.mkdir(dst_2)

for cls in os.listdir(src):
    cls_path=os.path.join(src,cls)
    dst_cls_path=os.path.join(dst_1,cls)
    total=0
    if not os.path.isdir(dst_cls_path):
        os.mkdir(dst_cls_path)
    for fname in os.listdir(cls_path):
        total+=1
    cut_at=int(np.floor(total*ratio))
    count=0
    for fname in os.listdir(cls_path):
        count+=1
        if count==cut_at:
            dst_cls_path=os.path.join(dst_2,cls)
            if not os.path.isdir(dst_cls_path):
                os.mkdir(dst_cls_path)
        shutil.copy(os.path.join(cls_path,fname), os.path.join(dst_cls_path,fname))
        
print 'finished'    