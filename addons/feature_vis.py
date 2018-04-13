import os,sys
import numpy as np
import cv2
import caffe
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def printRow(fs,record):
    for n in range(len(record)):
        fs.write('%f'%record[n])
        if n<len(record)-1:
            fs.write(',')
    fs.write('\n')

def preprocess(input,mode):
    output=[]
    if mode==1:
        output=np.zeros((3,input.shape[0],input.shape[1]),dtype=np.float32)
        gs=cv2.cvtColor(input,cv2.COLOR_BGR2GRAY)
        sX=cv2.Sobel(gs,cv2.CV_32F,1,0,ksize=3)
        sY=cv2.Sobel(gs,cv2.CV_32F,0,1,ksize=3)
        output[0,:,:]=sX
        output[1,:,:]=sY
        output[2,:,:]=gs.astype(float)
    elif mode==2:
        output=np.zeros((1,input.shape[0],input.shape[1]),dtype=np.float32)
        gs=cv2.cvtColor(input,cv2.COLOR_BGR2GRAY)
        output[0,:,:]=cv2.Laplacian(gs,cv2.CV_32F)
    elif mode==3:
        output=np.zeros((1,input.shape[0],input.shape[1]),dtype=np.float32)
        output[0,:,:]=cv2.cvtColor(input,cv2.COLOR_BGR2GRAY)    
    return output

predictions=[]
gt=[]
gt_total={}

model_def=sys.argv[1]
weights=sys.argv[2]
filelist=sys.argv[3]
output_cls_lbl_at=0
root_path=''
output_filename='output'
mode=0
vtype='tsne'
o_cls=[]
if len(sys.argv)>4:
    root_path=sys.argv[4]
if len(sys.argv)>5:
    output_filename=sys.argv[5]
if len(sys.argv)>6:
    lbls=sys.argv[6].split(',')
    output_cls_lbl_at=[]
    for lbl in lbls:
        output_cls_lbl_at.append(int(lbl))
    
if len(sys.argv)>7:
    vtype=sys.argv[7]

if len(sys.argv)>8:
    ocs=sys.argv[8].split(',')
    for oc in ocs:
        o_cls.append(int(oc))

if len(sys.argv)>9:
    if sys.argv[9]=='multich':
        mode=1
        
        

caffe.set_mode_gpu()
caffe.set_device(0)
net=caffe.Net(model_def,caffe.TEST,weights=weights)
in_=net.inputs[0]
shape=net.blobs[in_].data.shape
print shape
print net.outputs[0]

#outputp=open('%s.csv'%output_filename,'wb')
outputp=[]
count=0
input=np.zeros(shape,dtype=np.float32)
nOutput=len(net.outputs)
tsne_m=TSNE(n_components=2,init='pca')
pca_m=PCA(n_components=2)
lda_m=LDA(n_components=2)
features=[]
total=0
for i in range(nOutput):
    node_name=net.outputs[i].replace('/','_')
    f=open('%s_%s.csv'%(output_filename,node_name),'wb')
    outputp.append(f)
with open(filelist,'r') as f:
    for ln in f:
        line=ln.rstrip('\n')
        fields=line.split(',')
        if len(fields)==1:
            fields=line.split(' ')
        labels=fields[1].split(';')
        tlbl=int(labels[output_cls_lbl_at[0]])
        if o_cls!=[]:
            if tlbl not in o_cls:
                continue
        total+=1
        
count_row=0
lbls=[]
cc=0
print o_cls
with open(filelist,'r') as f:
    for ln in f:
        line=ln.rstrip('\n')
        fields=line.split(',')
        if len(fields)==1:
            fields=line.split(' ')
        labels=fields[1].split(';')
        tlbl=int(labels[output_cls_lbl_at[0]])
        if o_cls!=[]:
            if tlbl not in o_cls:
                continue
        
        filepath=os.path.join(root_path,fields[0].replace('\\','/'))
        img=cv2.imread(filepath)
        feed_input=cv2.resize(img,(shape[3],shape[2]))
        x=None
        if mode>0:
            x=preprocess(feed_input, mode)
        else:
            feed_input=feed_input[:,:,(2,1,0)]
            x=feed_input.swapaxes( 0, 2).swapaxes(1,2)
        input[0,:,:,:]=x
        
        y=net.forward_all(**{net.inputs[0]:input})
        i=0
        for i in range(nOutput):
            label=int(labels[output_cls_lbl_at[i]])
            feature=y[net.outputs[i]]
            if features==[]:
                for j in range(nOutput):
                    features.append(np.zeros((total,feature.shape[1]),dtype=feature.dtype))
                    lbls.append([])
            features[i][count_row,:]=feature[0,:]
            lbls[i].append(label)
        count_row+=1
        if count_row%100==0:
            print 'extracted %d/%d data'%(count_row,total)


for i in range(nOutput):
    #print features[i]
    l_feature=None
    if vtype=='pca':
        l_feature=pca_m.fit_transform(features[i])
    elif vtype=='lda':
        l_feature=lda_m.fit_transform(features[i])
    else:
        l_feature=tsne_m.fit_transform(features[i])
    #l_feature=pca_m.fit_transform(features[i])
    for n in range(total):
        prow=np.zeros(3,dtype=np.float32)
        prow[0]=l_feature[n,0]
        prow[1]=l_feature[n,1]
        prow[2]=lbls[i][n]
        printRow(outputp[i], prow)
    
    print l_feature.shape
for i in range(nOutput):
    outputp[i].close()