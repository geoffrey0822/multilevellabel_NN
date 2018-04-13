import os,sys
import numpy as np
import caffe
import xml.etree.ElementTree as ET

def append_multilabel(txn,id,key,labels):
    datum=caffe.proto.caffe_pb2.Datum()
    datum.channels=1
    datum.height=labels.shape[1]
    datum.width=labels.shape[2]
    datum.data=labels.tobytes()
    datum.label=0
    str_id='{:08}'.format(id)
    txn.put(str_id.encode('ascii'),datum.SerializeToString())
    return id+1
    pass

def create_multilabel(fname,nlabel, ndata):
    map_size=nlabel*ndata*4*10
    env=lmdb.open(fname,map_size=map_size)
    return env.begin(write=True)

def append_roi(txn,id,rois):
    datum=caffe.proto.caffe_pb2.Datum()
    datum.channels=1
    datum.height=rois.shape[1]
    datum.width=rois.shape[2]
    datum.data=rois.tobytes()
    datum.label=0
    str_id='{:08}'.format(id)
    txn.put(str_id.encode('ascii'),datum.SerializeToString())
    return id+1

def create_roi(fname,ndata,datasize):
    map_size=ndata*datasize*4*10
    env=lmdb.open(fname,map_size=map_size)
    return env.begin(write=True)

def append_general(txn,id,data,label):
    datum=caffe.proto.caffe_pb2.Datum()
    datum.channels=data.shape[0]
    datum.height=data.shape[1]
    datum.width=data.shape[2]
    datum.data=data.tobytes()
    datum.label=int(label)
    str_id='{:08}'.format(id)
    txn.put(str_id.encode('ascii'),datum.SerializeToString())
    return id+1

def create_general(fname,ndata,datasize,bytesize):
    map_size=ndata*datasize*bytesize*10
    env=lmdb.open(fname,map_size=map_size)
    return env.begin(write=True)

def readXML(fpath):
    tree=ET.parse(fpath)
    root=tree.getroot()
    img1_path=root.find('img1').text
    img2_path=root.find('img2').text
    
    lbl_elems=root.finall('label')
    labels=np.zeros((1,len(lbl_elems)),np.int32)
    i=0
    for lbl_elem in lbl_elems:
        labels[0,i]=int(lbl_elem.text)
        
    obj_elems=root.findall('object')
    bndboxs=np.zeros((len(obj_elems),4),np.float32)
    obj_labels=np.zeros((1,len(obj_elems)),np.int32)
    i=0
    for obj_elem in obj_elems:
        bndbox_str=obj_elem.find('roi').text.split(',')
        bndbox=[int(bndbox_str[0]),int(bndbox_str[1]),int(bndbox_str[2]),int(bndbox_str[3])]
        bndboxs[i,:]=bndbox
        obj_labels[0,i]=int(obj_elem.find('label').text)
        i+=1
    return [img1_path,img2_path],labels,bndboxs,obj_labels

def create_dataset(annotation_path,exp_path,single_file=False):
    if not single_file:
        
        for annotation in os.listdir(annotation_path):
            xml_path=os.path.join(annotation_path,annotation)
            image_pair,labels,rois,obj_labels=readXML(xml_path)
    else:
        pass
            