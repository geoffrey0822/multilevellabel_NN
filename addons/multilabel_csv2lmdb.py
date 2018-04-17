import os,sys
import numpy as np
import cv2
import time
import caffe
from caffe.proto import caffe_pb2
from numpy import uint8
import lmdb

csv_filepath=sys.argv[1]
img_root=sys.argv[2]
dst=sys.argv[3]
filename=sys.argv[4]
if not os.path.isdir(dst):
    os.mkdir(dst)

version=1
if len(sys.argv)>5:
    version=int(sys.argv[5])
size_str=[]
size=[]
if len(sys.argv)>6:
    size_str=sys.argv[6].split(',')
    size=[3,int(size_str[0]),int(size_str[1])]
    if len(size_str)>2:
        size=[int(size_str[0]),int(size_str[1]),int(size_str[2])]

padded_size=[]
if len(sys.argv)>8:
    padded_size_str=sys.argv[8].split(',')
    padded_size=[int(padded_size_str[0]),int(padded_size_str[1])]

N=1000
X=[]
Y=[]

total_data=0
total_bytes=0
nlabel=0
data_size=0
label_size=0
with open(csv_filepath,'r') as f:
    for ln in f:
        line=ln.rstrip('\n')
        total_data+=1
        fields=line.split(',')
        img_path=os.path.join(img_root,fields[0].replace('\\','/'))
        labels=fields[1].split(';')
        if nlabel==0:
            nlabel=len(labels)
        if(size==[] and padded_size==[]):
            img=cv2.imread(img_path)
            bsize=img.nbytes
            data_size+=bsize
            bsize+=4*nlabel
            label_size+=4*nlabel
            total_bytes+=bsize
            
        if total_data%1000==0:
            print 'scanned %d data'%total_data
            
print 'total %d data'%total_data
print 'number of label per data: %d'%nlabel

N=total_data

if size==[] and padded_size==[]:
    max_size=total_bytes*2
elif padded_size==[]:
    #X=np.zeros((N,size[0],size[1],size[2]),dtype=uint8)
    #Y=np.zeros((N,nlabel),dtype=np.int32)
    #max_size=(X.nbytes+Y.nbytes)*2;
    max_size=N*size[0]*size[1]*size[2]
    data_size=N*size[0]*size[1]*size[2]
    max_size+=N*4*nlabel
    label_size=N*4*nlabel
    max_size*=2
else:
    max_size=N*size[0]*padded_size[0]*padded_size[1]
    data_size=N*size[0]*padded_size[0]*padded_size[1]
    max_size+=N*4*nlabel
    label_size=N*4*nlabel
    max_size*=2
    
print 'the size of lmdb: %.2f mb'%(float(max_size)/1024/1024)

X=[]
Y=[]

#datum.channels = X.shape[1]
#datum.height=X.shape[2]
#datum.width=X.shape[3]
if version==2:
    print 'generating version 2\'s LMDB...'
    output_path=os.path.join(dst,filename)
    env=lmdb.open(output_path,map_size=max_size)
    with env.begin(write=True) as txn:
        with open(csv_filepath,'r') as f:
            i=0
            for ln in f:
                line=ln.rstrip('\n')
                total_data+=1
                fields=line.split(',')
                img_path=os.path.join(img_root,fields[0].replace('\\','/'))
                labels_str=fields[1].split(';')
                img=cv2.imread(img_path)
                if size!=[]:
                    img=cv2.resize(img,(size[1],size[2]))
                if padded_size!=[]:
                    top=int(padded_size[0]-size[1]/2)
                    bottom=top
                    left=int(padded_size[1]-size[2]/2)
                    right=left
                    img=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,[0,0,0])
                caffe_data=img[:,:,(2,1,0)]
                caffe_data=caffe_data.swapaxes(0,2).swapaxes(1,2)
                datum=caffe_pb2.MultiLabelDatum()
                if size==[]:
                    datum.channels=caffe_data.shape[0]
                    datum.height=caffe_data.shape[1]
                    datum.width=caffe_data.shape[2]
                elif padded_size==[]:
                    datum.channels=size[0]
                    datum.height=size[1]
                    datum.width=size[2]
                else:
                    datum.channels=size[0]
                    datum.height=padded_size[0]
                    datum.width=padded_size[1]
                datum.data=caffe_data.tobytes()
                labels=[]
                for j in range(nlabel):
                    labels.append(int(labels_str[j]))
                
                datum.labels.extend(labels)
                str_id='{:08}'.format(i)
                txn.put(str_id.encode('ascii'),datum.SerializeToString())
                i+=1
                if i%1000==0:
                    print 'added %d data into LMDB'%i
else:
    label_lmdb='label_%s'%filename
    data_lmdb='data_%s'%filename
    output_label=os.path.join(dst,label_lmdb)
    output_data=os.path.join(dst,data_lmdb)
    env_data=lmdb.open(output_data,map_size=data_size*2)
    env_label=lmdb.open(output_label,map_size=label_size*10)
    with env_data.begin(write=True) as txn_data:
        with env_label.begin(write=True) as txn_label:
            with open(csv_filepath,'r') as f:
                i=0
                for ln in f:
                    line=ln.rstrip('\n')
                    total_data+=1
                    fields=line.split(',')
                    img_path=os.path.join(img_root,fields[0].replace('\\','/'))
                    labels_str=fields[1].split(';')
                    img=cv2.imread(img_path)
                    if size!=[]:
                        img=cv2.resize(img,(size[1],size[2]))
                    if padded_size!=[]:
                        top=int(padded_size[0]-size[1]/2)
                        bottom=top
                        left=int(padded_size[1]-size[2]/2)
                        right=left
                        img=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,[0,0,0])
                    caffe_data=img[:,:,(2,1,0)]
                    caffe_data=caffe_data.swapaxes(0,2).swapaxes(1,2)
                    datum=caffe_pb2.Datum()
                    lbl_datum=caffe_pb2.Datum()
                    if size==[]:
                        datum.channels=caffe_data.shape[0]
                        datum.height=caffe_data.shape[1]
                        datum.width=caffe_data.shape[2]
                    elif padded_size==[]:
                        datum.channels=size[0]
                        datum.height=size[1]
                        datum.width=size[2]
                    else:
                        datum.channels=size[0]
                        datum.height=padded_size[0]
                        datum.width=padded_size[1]
                    datum.data=caffe_data.tobytes()
                    labels=[]
                    lbl_datum.channels=nlabel
                    for j in range(nlabel):
                        labels.append(float(labels_str[j]))
                    lbl_datum.channels=nlabel
                    lbl_datum.width=1
                    lbl_datum.height=1
                    lbl_datum.float_data.extend(labels)
                    str_id='{:08}'.format(i)
                    txn_data.put(str_id.encode('ascii'),datum.SerializeToString())
                    txn_label.put(str_id.encode('ascii'),lbl_datum.SerializeToString())
                    i+=1
                    if i%1000==0:
                        print 'added %d data into LMDB'%i

print 'finished'