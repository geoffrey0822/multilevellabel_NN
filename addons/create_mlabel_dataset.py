import os,sys,time
import numpy as np
import shutil

total_class_count=0

def exportData(root,root_name,fst,fsv,cls_map,dst_path,ntrain):
    global total_class_count
    keys=cls_map.keys()
    count=0
    millis=int(round(time.time()*1000))
    total_data=0
    itrain=ntrain
    if root_name in keys:
        total_class_count+=1
        print 'processing on %s %d/%d'%(root_name,total_class_count,len(keys))
        if ntrain<=1:
            for file in os.listdir(root):
                total_data+=1
            total_data+=1
            if ntrain<1:
                itrain=total_data*ntrain 
            else:
                itrain=total_data
        
    for file in os.listdir(root):
        new_path=os.path.join(root,file)
        if os.path.isdir(new_path):
            exportData(new_path, file, fst,fsv, cls_map,dst_path,ntrain)
        else:
            if root_name in keys:
                count+=1
                img_name='%d_%d.jpg'%(millis,count)
                img_path=os.path.join(dst_path,img_name).replace('\\','/')
                shutil.copy(os.path.join(root,file), img_path)
                if count<itrain:
                    fst.write('%s,%s\n'%(img_name,cls_map[root_name]))
                else:
                    fsv.write('%s,%s\n'%(img_name,cls_map[root_name]))

root=sys.argv[1]
dst_path=sys.argv[2]
output_name=sys.argv[3]
base_index=sys.argv[4]
lv_index=sys.argv[5]
itype=sys.argv[6]
ntrain=0
if itype=='quantity':
    ntrain=int(sys.argv[7])
else:
    ntrain=float(sys.argv[7])

if not os.path.isdir(dst_path):
    os.mkdir(dst_path)
    
img_path=os.path.join(dst_path,'imgs')

if not os.path.isdir(img_path):
    os.mkdir(img_path)
    
output_tpath=os.path.join(dst_path,'train_%s'%output_name)
output_vpath=os.path.join(dst_path,'val_%s'%output_name)

cls_map={}
with open(base_index,'r') as f:
    for ln in f:
        line=ln.rstrip('\n')
        fields=line.split(':')
        cls_map[fields[0]]=0
        
with open(lv_index,'r') as f:
    cls_keys=cls_map.keys()
    for ln in f:
        line=ln.rstrip('\n')
        fields=line.split(':')
        if fields[0] in cls_keys:
            cls_map[fields[0]]=fields[1]


with open(output_tpath,'wb') as ft:
    with open(output_vpath,'wb') as fv:
        exportData(root, '', ft,fv, cls_map, img_path,ntrain)         
#print cls_map
print 'finished'