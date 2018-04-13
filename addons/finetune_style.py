import os,sys
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

prototxt_file=sys.argv[1]
to_prototxt_file=sys.argv[2]
hold_layers=sys.argv[3].split(',')

model=caffe_pb2.NetParameter()

with open(prototxt_file,'r') as f:
    txtf.Merge(f.read(),model)
    f.close()

active_layers={}
i=0
for layer in model.layer:
    if len(layer.param)>0:
        isActive=False
        for param in layer.param:
            isActive=param.lr_mult>0 or param.decay_mult>0
            if isActive:
                break
        active_layers[layer.name]=i
    i+=1
        
print active_layers
print model.layer[active_layers['cls']]
for lname in active_layers.keys():
    if lname in hold_layers:
        continue
    nParam=len(model.layer[active_layers[lname]].param)
    for n in range(nParam):
        model.layer[active_layers[lname]].param[n].lr_mult=0
        model.layer[active_layers[lname]].param[n].decay_mult=0

with open(to_prototxt_file,'w') as f:
    f.write(str(model))
    f.close()

print 'finished'