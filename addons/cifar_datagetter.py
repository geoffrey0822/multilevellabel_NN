import os,sys
import numpy as np
import time
import cv2

def unpickle(file):
    import cPickle
    dict={}
    with open(file,'rb') as fo:
        dict=cPickle.load(fo)
    return dict

src=sys.argv[1]
src_like=sys.argv[2]
meta=sys.argv[3]
dst=sys.argv[4]
version=sys.argv[5]
truelbl_at=0
if version=='100':
    truelbl_at=int(sys.argv[6])

if not os.path.isdir(dst):
    os.mkdir(dst)
    
    
meta_data=unpickle(meta)
lbl_names=[]
super_class=[]
if version=='10':
    lbl_names= meta_data['label_names']
else:
    lbl_names=meta_data['fine_label_names']
    super_class=meta_data['coarse_label_names']
    
print lbl_names
for file in os.listdir(src):
    if file.startswith(src_like):
        print 'doing %s...'%file
        file_path=os.path.join(src,file)
        if os.path.isdir(file_path):
            continue
        data=unpickle(os.path.join(src,file))
        labels=[]
        if version=='10':
            labels=data['labels']
        else:
            labels=data['fine_labels']
        nRow=len(labels)
        millis = int(round(time.time() * 1000))
        for n in range(nRow):
            lbl_name=lbl_names[labels[n]]
            dst_path=os.path.join(dst,lbl_name)
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            img_data=data['data'][n]
            img_data=np.reshape(img_data,(3,32,32))
            img_data=img_data.swapaxes(0,1).swapaxes(1,2)[:,:,(2,1,0)]
            ofname='%d_%d.jpg'%(millis,n)
            cv2.imwrite(os.path.join(dst_path,ofname),img_data)
            if n>0 and n%1000==0:
                print("processed %d"%n)

print 'finished'