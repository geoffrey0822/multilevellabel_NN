import os,sys
import numpy as np
import cv2
import time
import shutil

key_tree=[]

def loadKeys(file_path):
    global key_tree
    with open(file_path,'r') as f:
        for ln in f:
            line=ln.rstrip('\n')
            fields=line.split(',')
            lv=0
            for field in fields:
                cells=field.split(':')
                if key_tree==[] or len(key_tree)<=lv:
                    key_tree.append({})
                if cells[1] not in key_tree[lv]: 
                    key_tree[lv][cells[1]]=int(cells[0])
                lv+=1

def doNested(src,super_keys,dst,alias_dst,outputf):
    global key_tree
    for fname in os.listdir(src):
        file_path=os.path.join(src,fname)
        if os.path.isdir(file_path):
            new_keys=[]
            for i in range(len(super_keys)):
                new_keys.append(super_keys[i])
            new_keys.append(fname)
            doNested(file_path,new_keys,dst,alias_dst,outputf)
            print new_keys
        else:
            millis=int(round(time.time()*1000))
            new_file_path=os.path.join(dst,'%ld_%s'%(millis,fname))
            new_afile_path=os.path.join(alias_dst,'%ld_%s'%(millis,fname))
            key_str=''
            #print super_keys
            for i in range(len(super_keys)):
                key_str+=str(key_tree[i][super_keys[i]])
                if i<len(super_keys)-1:
                    key_str+=';'
            if not os.path.isdir(dst):
                os.mkdir(dst)
            shutil.copy(file_path, new_file_path)
            outputf.write('%s,%s\n'%(new_afile_path,key_str))

src=sys.argv[1]
dst=sys.argv[2]
labels=sys.argv[3] # class_idx_1:label_1,class_idx_2:label_2, ...
dst_db='dataset.csv'
if len(sys.argv)>4:
    dst_db=sys.argv[4]

dst_alias=os.path.basename(os.path.normpath(dst))

if not os.path.isdir(dst):
    os.mkdir(dst)
loadKeys(labels)
print key_tree

with open(dst_db,'w') as outputf:
    doNested(src, [], dst,dst_alias, outputf)

print 'finished'
