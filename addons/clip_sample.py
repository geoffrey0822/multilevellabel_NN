import os,sys
import numpy as np
import shutil

src=sys.argv[1]
dst=sys.argv[2]

for cls in os.listdir(src):
    src_cls_path=os.path.join(src,cls)
    dst_cls_path=os.path.join(dst,cls)
    if not os.path.isdir(dst_cls_path):
        os.mkdir(dst_cls_path)
    for fname in os.listdir(src_cls_path):
        shutil.copy(os.path.join(src_cls_path,fname), dst_cls_path)
        break
    
print 'finish'