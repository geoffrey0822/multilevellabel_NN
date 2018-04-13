import caffe
import os,sys
import cv2
import numpy as np

def getFeature(model,input,lname):
    transformer=caffe.io.Transformer({'data':model.blobs['data'].data.shape})
    transformer.set_channel_swap('data', (2,0,1))
    #model.blobs['data'].data[...]=input
    #iinput=input.astype('float')/225.
    iinput=input[:,:,(2,1,0)]
    iinput=np.swapaxes(iinput, 0, 2)
    iinput=np.swapaxes(iinput, 1, 2)
    iiinput=np.zeros((1,input.shape[2],input.shape[0],input.shape[1]),dtype=np.float32)
    iiinput[0,:,:,:]=iinput
    #model.blobs['data'].data[...]=transformer.preprocess('data', iinput)
    model.blobs['data'].data[...]=iiinput
    return model.forward()[lname]

def procDir(model,src,exp,limit=0):
    maxLayer=len(model._layer_names)
    outputName=model._layer_names[maxLayer-1]
    print outputName
    if os.path.isfile(exp):
        os.remove(exp)
    with open(exp,'a') as ffile:
        
        i=0
        for imgf in os.listdir(src):
            img_path=os.path.join(src,imgf)
            img=cv2.imread(img_path)
            if img is None:
                continue
            newDim=(model.blobs['data'].data.shape[2],model.blobs['data'].data.shape[3])
            feat=getFeature(model, cv2.resize(img,newDim),outputName)
            np.savetxt(ffile,feat,delimiter=',')
            i+=1
            if i%100==0:
                print i
            if limit>0 and i>=limit:
                break
        ffile.close()
    print 'finish'

prototxt_file=sys.argv[1]
param_file=sys.argv[2]
data_dir=sys.argv[3]
exp_file=sys.argv[4]
limit=0
if len(sys.argv)>5:
    limit=int(sys.argv[5])

caffe.set_device(0)
caffe.set_mode_gpu()
net=caffe.Net(prototxt_file,param_file,caffe.TEST)
procDir(net,data_dir, exp_file,limit)
