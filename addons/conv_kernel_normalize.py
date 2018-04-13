import os,sys
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

prototxt_file=sys.argv[1]
to_prototxt_file=sys.argv[2]
kernel_size=int(sys.argv[3])

model=caffe_pb2.NetParameter()

with open(prototxt_file,'r') as f:
    txtf.Merge(f.read(),model)
    f.close()

active_layers=[]
i=0
for layer in model.layer:
    if layer.type=='Convolution':
        conv_param=layer.convolution_param
        if conv_param.kernel_size[0]>kernel_size:
            active_layers.append(i)
    i+=1
        
for idx in active_layers:
    opad=model.layer[idx].convolution_param.pad
    model.layer[idx].convolution_param.kernel_size[0]=kernel_size
    if opad!=[] and opad[0]>0:
        model.layer[idx].convolution_param.pad[0]=(kernel_size-1)/2
        

with open(to_prototxt_file,'w') as f:
    f.write(str(model))
    f.close()

print 'finished'