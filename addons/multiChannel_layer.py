import os,sys
import numpy as np
import caffe
import json

class SlidingFill_Layer(caffe.Layer):
    def setup(self,bottom,top):
        params=json.loads(self.param_str)
        self.imgSize=params['image_size']
        self.winSize=params['kernel_size']
        self.stride=params['stride']
        assert(len(self.imgSize.shape)==2 and 
               len(self.winSize.shape)==2 and 
               len(self.stride.shape)==2)
        self.boxes=self.computeBBox(self.imgSize,self.winSize,self.stride)
    def reshape(self,bottom,top):
        top[0].reshape(1)
        pass
    def forward(self,bottom,top):
        top[0].data[...]=self.boxes
    def backward(self,bottom,top):
        pass
    def computeBBox(self,rawSize,winSize,stride):
        n_col=np.int32(np.floor(np.float32(rawSize[1]-winSize[1])/stride[1]+1))
        n_row=np.int32(np.floor(np.float32(rawSize[0]-winSize[0])/stride[0]+1))
        bboxes=np.zeros((n_row,n_col),np.int32)
        for row in range(n_row):
            for col in range(n_col):
                y=row*stride[0]
                x=col*stride[1]
                y_max=y+winSize[0]
                x_max=x+winSize[1]
                if y_max>=rawSize[0]:
                    y=rawSize[0]-winSize[0]-1
                bboxes[row*n_col+col,:]=np.int32((y,x,y_max,x_max))
        return bboxes
    
class Projection_Layer(caffe.Layer):
    def setup(self,bottom,top):
        assert(len(bottom.shape)==2)
        pass
    def reshape(self,bottom,top):
        boxes=bottom[1].data
        feat_map=bottom[0].data
        nBatch=boxes.shape[0]
        nCh=feat_map.shape[1]
        width=feat_map.shape[3]
        pass
    def forward(self,bottom,top):
        if self.phase==caffe.TEST:
            print 'do'
        else:
            print 'nothing'
    def backward(self,top,propagate_down,bottom):
        pass