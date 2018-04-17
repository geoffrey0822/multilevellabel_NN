import os,sys
import numpy as np

def getNestedClasses(root,depth,max_depth,neighbor_length=0):
    new_depth=depth+1
    classes={}
    total_cls=0
    super_cls=0
    for cls in os.listdir(root):
        if not os.path.isdir(os.path.join(root,cls)):
            continue
        if new_depth<max_depth:
            pack=getNestedClasses(os.path.join(root,cls), new_depth, max_depth,neighbor_length+total_cls)
            total_cls+=pack[1]
            classes[cls]=[neighbor_length+super_cls,pack[0]]
        else:
            classes[cls]=[neighbor_length+total_cls]
            total_cls+=1
        super_cls+=1
    return [classes,total_cls]

def exportLevelLabel(fs,level_keys,target_lv,lv=0,accum_label=''):
    new_lv=lv+1
    if lv==target_lv:
        for key in level_keys.keys():
            lbl='%s%d'%(accum_label,level_keys[key][0])
            fs.write('%s:%s\n'%(key,lbl))
    else:
        for key in level_keys.keys():
            new_accum_label='%d;'%level_keys[key][0]
            exportLevelLabel(fs, level_keys[key][1], target_lv, new_lv,new_accum_label)
            
def exportLevelLabel_noaccum(fs,level_keys,target_lv,lv=0):
    new_lv=lv+1
    if lv==target_lv:
        for key in level_keys.keys():
            fs.write('%s:%d\n'%(key,level_keys[key][0]))
    else:
        for key in level_keys.keys():
            exportLevelLabel_noaccum(fs, level_keys[key][1], target_lv, new_lv)

def getBaseMap(level_keys,target_lv,map,lv=0):
    new_lv=lv+1
    if lv==target_lv:
        for key in level_keys.keys():
            map.append(0)
    else:
        for key in level_keys.keys():
            getBaseMap(level_keys[key][1], target_lv, map, new_lv)

def exportIndexMap(fs,level_keys,base_map,target_lv,lv=0,parent_idx=-1):
    new_lv=lv+1
    if lv==target_lv:
        for key in level_keys.keys():
            base_map[level_keys[key][0]]=parent_idx
    elif new_lv==target_lv:
        for key in level_keys.keys():
            idx=level_keys[key][0]
            exportIndexMap(fs, level_keys[key][1], base_map, target_lv, new_lv, idx)
        for idx in base_map:
            fs.write('%d\n\n'%idx)
    
root=sys.argv[1]
dst=sys.argv[2]
filename=sys.argv[3]
nlevel=int(sys.argv[4])

label_file=''
if len(sys.argv)>6:
    label_file=sys.argv[6]

if not os.path.isdir(dst):
    os.mkdir(dst)
    
dst_label=os.path.join(dst,'label_%s'%filename)
dst_blabel=os.path.join(dst,'base_label_%s'%filename)
dst_indexmap=os.path.join(dst,filename)
level_keys=getNestedClasses(root, 0, nlevel)[0]
with open(dst_label,'wb') as f:
    for n in range(nlevel):
        print 'exporting for level %d'%(n+1)
        exportLevelLabel(f, level_keys, n)
        f.write('\n')
        
with open(dst_blabel,'wb') as f:
    exportLevelLabel_noaccum(f, level_keys, nlevel-1)
    
with open(dst_indexmap,'wb') as f:
    for n in range(nlevel):
        if n==0:
            continue
        base_map=[]
        getBaseMap(level_keys, n,base_map)
        print base_map
        print 'No. of class in level %d: %d'%(nlevel,len(base_map))
        exportIndexMap(f,level_keys,base_map,n)
        print 'after assigned'
        print base_map
        
print 'finished'
