import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import os,sys
import numpy as np

digit_model=sys.argv[1]
dst=sys.argv[2]
lr_rate=1e-2
lr_policy='step'
lr_stepratio=0.3
epoch=1
val_at_epoch=1
t_ndata=0.0
v_ndata=0.0
t_nbatch=1.0
v_nbatch=1.0
snapshot=1
snapshot_prefix='snapshot'
gamma=1e-1
ch=3
width=0
height=0
stype='SGD'

if len(sys.argv)>3:
    lr_rate=float(sys.argv[3])
if len(sys.argv)>4:
    lr_policy=sys.argv[4]
if len(sys.argv)>5:
    lr_stepratio=float(sys.argv[5])
if len(sys.argv)>6:
    epoch=float(sys.argv[6])
if len(sys.argv)>7:
    val_at_epoch=float(sys.argv[7])
if len(sys.argv)>8:
    t_ndata=float(sys.argv[8])
if len(sys.argv)>9:
    t_nbatch=float(sys.argv[9])
if len(sys.argv)>10:
    v_ndata=float(sys.argv[10])
if len(sys.argv)>11:
    v_nbatch=float(sys.argv[11])
if len(sys.argv)>12:
    snapshot=float(sys.argv[12])
if len(sys.argv)>13:
    width=int(sys.argv[13])
if len(sys.argv)>14:
    height=int(sys.argv[14])
if len(sys.argv)>15:
    ch=int(sys.argv[15])
if len(sys.argv)>16:
    stype=sys.argv[16]
    
net=caffe_pb2.NetParameter()
trainval_net=caffe_pb2.NetParameter()
deploy_net=caffe_pb2.NetParameter()
solver=caffe_pb2.SolverParameter()

deploy_net.input.append('data')
deploy_net.input_shape.add(dim=[1,ch,height,width])

with open(digit_model) as f:
    s=f.read()
    txtf.Merge(s,net)
    
if not os.path.isdir(dst):
    os.mkdir(dst)

for l in net.layer:
    if len(l.include)==1:
        if l.include[0].stage[0]=='deploy':
            deploy_newlayer=deploy_net.layer.add()
            deploy_newlayer.CopyFrom(l)
            del deploy_newlayer.include[:]
        elif l.include[0].stage[0]=='train':
            tv_newlayer=trainval_net.layer.add()
            tv_newlayer.CopyFrom(l)
            del tv_newlayer.include[:]
            tv_newlayer.include.add(phase=caffe_pb2.TRAIN)
        elif l.include[0].stage[0]=='val':
            tv_newlayer=trainval_net.layer.add()
            tv_newlayer.CopyFrom(l)
            del tv_newlayer.include[:]
            tv_newlayer.include.add(phase=caffe_pb2.TEST)
    elif len(l.exclude)==1:
        if l.exclude[0].stage[0]=='deploy':
            tv_newlayer=trainval_net.layer.add()
            tv_newlayer.CopyFrom(l)
            del tv_newlayer.exclude[:]
        elif l.exclude[0].stage[0]=='train':
            deploy_newlayer=deploy_net.layer.add()
            deploy_newlayer.CopyFrom(l)
            del deploy_newlayer.exclude[:]
            tv_newlayer=trainval_net.layer.add()
            tv_newlayer.CopyFrom(l)
            del tv_newlayer.exclude[:]
            tv_newlayer.include.add(phase=caffe_pb2.TEST)
        elif l.exclude[0].stage[0]=='val':
            deploy_newlayer=deploy_net.layer.add()
            deploy_newlayer.CopyFrom(l)
            del deploy_newlayer.exclude[:]
            tv_newlayer=trainval_net.layer.add()
            tv_newlayer.CopyFrom(l)
            del tv_newlayer.exclude[:]
            tv_newlayer.include.add(phase=caffe_pb2.TRAIN)
    else:
        deploy_newlayer=deploy_net.layer.add()
        deploy_newlayer.CopyFrom(l)
        tv_newlayer=trainval_net.layer.add()
        tv_newlayer.CopyFrom(l)
        
solver.base_lr=lr_rate
solver.lr_policy=lr_policy
solver.gamma=gamma
solver.max_iter=int(epoch*int(np.ceil(t_ndata/t_nbatch)))
solver.iter_size=1
solver.test_iter.append(int(np.ceil(v_ndata/v_nbatch)))
solver.test_interval=int(val_at_epoch*int(np.ceil(t_ndata/t_nbatch)))
solver.type=stype
if stype!='AdaGrad':
    solver.momentum=1e-1
solver.snapshot=int(snapshot*int(np.ceil(t_ndata/t_nbatch)))
solver.snapshot_prefix=snapshot_prefix
solver.solver_mode=1
solver.weight_decay=1e-5
solver.net='trainval.prototxt'
solver.stepsize=int(np.ceil(lr_stepratio*np.ceil(epoch*np.ceil(t_ndata/t_nbatch))))

trainval_path=os.path.join(dst,'trainval.prototxt')
deploy_path=os.path.join(dst,'deploy.prototxt')
solver_path=os.path.join(dst,'solver.prototxt')

with open(trainval_path,'w') as f:
    f.write(txtf.MessageToString(trainval_net))
    f.close()
        
with open(deploy_path,'w') as f:
    f.write(txtf.MessageToString(deploy_net))
    f.close()
    
with open(solver_path,'w') as f:
    f.write(txtf.MessageToString(solver))
    f.close()
            
print 'finished'