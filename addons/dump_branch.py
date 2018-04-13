import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import os,sys

base_model=sys.argv[1]
exp_filename=sys.argv[2]
diverge_layer=sys.argv[3]
nBranch=int(sys.argv[4])

net=caffe_pb2.NetParameter()
newNet=caffe_pb2.NetParameter()

with open(base_model) as f:
    s=f.read()
    txtf.Merge(s,net)

layerNames=[l.name for l in net.layer]
old_layers=net.layer
copy_layers=[]
prefix='dump'
past_name=''
afterDiverge=False
for l in old_layers:
    if afterDiverge:
        copy_layers.append(l)
    else:
        tmp=newNet.layer.add()
        tmp.CopyFrom(l)
    if l.name==diverge_layer:
        afterDiverge=True
        print 'ok'
        
for branch in range(nBranch):
    newPrefix='%s_%d'%(prefix,branch+1)
    if branch==0:
        newPrefix=''
    for l in copy_layers:
        tmp=newNet.layer.add()
        tmp.CopyFrom(l)
        new_name='%s/%s'%(newPrefix,l.name)
        new_top='%s/%s'%(newPrefix,l.top[0])
        if newPrefix=='':
            new_name=l.name
            new_top=l.top[0]
        tmp.name=new_name
        tmp.top[0]=l.top[0]
        if tmp.top[0]!='data':
            tmp.top[0]=new_top
        for bottom in range(len(l.bottom)):
            new_bottom='%s/%s'%(newPrefix,l.bottom[bottom])
            if newPrefix=='':
                new_bottom=l.bottom[bottom]
            tmp.bottom[bottom]=l.bottom[bottom]
            if tmp.bottom[bottom]!='data' and tmp.bottom[bottom]!='label':
                tmp.bottom[bottom]=new_bottom
        print l.name
        
with open(exp_filename,'w') as f:
        f.write(txtf.MessageToString(newNet))
        f.close()    
print 'finish'