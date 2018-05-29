import os,sys
import numpy as np
import cv2
import caffe
from sklearn.metrics import confusion_matrix

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
true_at=0
root_path=''
output_filename='output'
mode=0
rows=[]
extracted=0
if len(sys.argv)>4:
    root_path=sys.argv[4]
if len(sys.argv)>5:
    true_at=int(sys.argv[5])
if len(sys.argv)>6:
    output_filename=sys.argv[6]
if len(sys.argv)>7:
    if sys.argv[7]=='multich':
        mode=1
if len(sys.argv)>8:
    rr=sys.argv[8].split(',')
    for r in rr:
        rows.append(int(r))

#caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(0)
net=caffe.Net(model_def,caffe.TEST,weights=weights)
in_=net.inputs[0]
shape=net.blobs[in_].data.shape
print shape
print net.outputs[0]

outputp=open('%s.csv'%output_filename,'wb')
outputgt=open('%s_gt.csv'%output_filename,'wb')
count=0
input=np.zeros(shape,dtype=np.float32)
nextR=0
print rows
with open(filelist,'r') as f:
    for ln in f:
        line=ln.rstrip('\n')
        fields=line.split(',')
        labels=fields[1].split(';')
        label=labels[true_at]
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
        scores=y[net.outputs[0]]
        if count in rows:
            printRow(outputp,scores[0,:])
            print '%d saved'%count
            nextR+=1
            extracted+=1
            if extracted>=len(rows):
                break
        
        printRow(outputp,scores[0,:])
        gt_vec=np.zeros(len(scores[0,:]),np.float)
        
        #printRow(outputp, scores)
        #printRow(outputgt, 1)
        count+=1
        if count%100==0:
            print 'predicted for %d'%count
        #predictions.append(np.argmax(scores))
        cls=int(label)
        if cls not in gt_total.keys():
            gt_total[cls]=1
        else:
            gt_total[cls]+=1
        gt.append(cls)
        gt_vec[cls]=1
        printRow(outputgt,gt_vec)
        #print scores
        #print np.argmax(scores)
        #break
        
outputp.close()
outputgt.close()
print 'finish'
exit()

outputmt=open('%s_mat.csv'%output_filename,'wb')
cmat=confusion_matrix(gt,predictions)
for row in range(len(gt_total.keys())):
    printRow(outputmt, cmat[row])
outputmt.close()
print 'finished'