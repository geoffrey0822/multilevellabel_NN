import numpy as np
import lmdb
import caffe
from caffe.proto import caffe_pb2
import os,sys

lmdb_file=sys.argv[1]
checkData=int(sys.argv[2])

env=lmdb.open(lmdb_file,readonly=True)
with env.begin() as txn:
    cursor=txn.cursor()
    for key,value in cursor:
        datum=caffe_pb2.MultiLabelDatum()
        datum.ParseFromString(value)
    
        x_bytes=np.fromstring(datum.data,dtype=np.uint8)
        #print len(x_bytes)
        x=x_bytes.reshape(datum.channels,datum.height,datum.width)
        print key
        print datum.labels