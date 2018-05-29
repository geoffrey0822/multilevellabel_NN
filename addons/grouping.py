import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def printRow(fs,record):
    for n in range(len(record)):
        fs.write('%f'%record[n])
        if n<len(record)-1:
            fs.write(',')
    fs.write('\n')

labels=[]
score_file=sys.argv[1]
index_map_file=sys.argv[2]
gt_file=sys.argv[3]
label_file=sys.argv[4]
out_file=sys.argv[5]

isGroupped=False
if len(sys.argv)>6:
    isGroupped=True
out_png_file='%s.png'%out_file

rooms_t0={}
rooms_t1={}
class_relations=[]
total_samples=[]
confuse2group=[]

cls_id=0

        
with open(index_map_file,'r') as f:
    for ln in f:
        line=ln.rstrip('\n')
        if len(line)==0:
            continue
        super_class=int(line)
        if super_class not in rooms_t0.keys():
            rooms_t0[super_class]=[]
        rooms_t0[super_class].append(cls_id)
        class_relations.append(super_class)
        total_samples.append(0)
        cls_id+=1
        labels.append('')
        
with open(label_file,'r') as f:
    for ln in f:
        line=ln.rstrip('\n')
        fields=line.split(':')
        labels[int(fields[1])]=fields[0]
        
print rooms_t0
dim=len(class_relations)
confusionMatrix=np.zeros((dim,dim),np.float)
outputf=open(out_file,'wb')
with open(score_file,'r') as f:
    with open(gt_file,'r') as gt_f:
        for ln in f:
            gt_ln=gt_f.next()
            line=ln.rstrip('\n')
            gt_line=gt_ln.rstrip('\n')
            fields=line.split(',')
            gt_fields=gt_line.split(',')
            scores=np.zeros(len(fields),np.float)
            gt_scores=np.zeros(len(gt_fields),np.float)
            i=0
            for field in fields:
                scores[i]=float(field)
                gt_scores[i]=float(gt_fields[i])
                i+=1
            class_id=np.argmax(scores)
            gt_class_id=np.argmax(gt_scores)
            confusionMatrix[gt_class_id,class_id]+=1

rects=[]            

for cls in range(dim):
    total=0
    for predict in range(dim):
        total+=confusionMatrix[cls,predict]
    for predict in range(dim):
        confusionMatrix[cls,predict]/=total
    printRow(outputf, confusionMatrix[cls,:])

if isGroupped:
    tmp_confusionMatrix=np.zeros((dim,dim),np.float)
    new_order=[]
    start_from=-0.5
    for room in rooms_t0.keys():
        for roommate in rooms_t0[room]:
            new_order.append(roommate)
        nMate=len(rooms_t0[room])
        rects.append(patches.Rectangle((start_from,start_from),nMate,nMate,linewidth=3,edgecolor='r',facecolor='none'))
        start_from+=nMate
    for cls in range(dim):
        group_confuse={}
        count=0
        for room in rooms_t0.keys():
            if room==class_relations[cls]:
                continue
            if room not in group_confuse.keys():
                group_confuse[room]=0
            for predict in rooms_t0[room]:
                group_confuse[room]+=confusionMatrix[cls,predict]
            for group in group_confuse.keys():
                group_confuse[group]=round(group_confuse[group],2)
        confuse2group.append(group_confuse)
    print confuse2group
#print confusionMatrix
outputf.close()
print labels 
fig,ax=plt.subplots()
im=ax.imshow(confusionMatrix,interpolation='nearest',aspect='auto')
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.setp(ax.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
for i in range(dim):
    for j in range(dim):
        text=ax.text(j,i,round(confusionMatrix[i,j]*100,1),ha='center',va='center',color='w',size=9)

for rect in rects:
    ax.add_patch(rect)
#rect=patches.Rectangle((9.5,14.5),5,5,linewidth=3,edgecolor='r',facecolor='none')
#ax.add_patch(rect)
fig.tight_layout()
plt.savefig(out_png_file)

remap_file='%s_remap.csv'%out_file
with open(remap_file,'wb') as map_f:
    for cls in range(dim):
        max_key=0
        maxVal=0
        group_info=confuse2group[cls]
        for group in group_info.keys():
            if group_info[group]>maxVal:
                maxVal=group_info[group]
                max_key=group
        map_f.write('%s,%d,%f\n'%(labels[cls],max_key,round(confusionMatrix[cls,cls]*100,2)))
print 'finished'