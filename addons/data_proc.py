import os,sys
import numpy as np
from shutil import copyfile


def exp_classes_restructed(src,dst):
    for cls in os.listdir(src):
        cls_path=os.path.join(src,cls)
        if not os.path.isdir(cls_path):
            continue
        exp_cls=os.path.join(dst,cls)
        if not os.path.isdir(exp_cls):
            os.mkdir(exp_cls)
        for data_dir in os.listdir(cls_path):
            data_path=os.path.join(cls_path,data_dir)
            if not os.path.isdir(data_path):
                continue
            for data in os.listdir(data_path):
                exp_path=os.path.join(exp_cls,data)
                copyfile(os.path.join(data_path,data),exp_path)
                

def exp_classes(annotation,src,dst):
    if annotation==None or annotation=='':
        exp_classes_restructed(src, dst)
        return
        
    with open(annotation) as f:
        line=f.readline().rstrip('\n')
        while line:
            line=f.readline().rstrip('\n')
            cols=line.split('\t')
            if cols==None or len(cols)<=1:
                continue
            exp_dir=os.path.join(dst,cols[1])
            if not os.path.isdir(exp_dir):
                os.mkdir(exp_dir)
            copyfile(os.path.join(src,cols[0]),os.path.join(exp_dir,cols[0]))
        f.close()

def exp_roi(annotation,src,dst):
    pass

src_dir=sys.argv[1]
mode=sys.argv[2]
dst_dir=sys.argv[3]
annotation=''
if len(sys.argv)>4:
    annotation=sys.argv[4]

if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)
    
if mode=='class':
    exp_classes(annotation, src_dir,dst_dir)
elif mode=='roi':
    exp_roi(annotation, src_dir,dst_dir)
else:
    print('invalid mode')
    
print('finished')