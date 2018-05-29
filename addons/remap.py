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

def compute_upperTriangle_sum(cmat):
    dim=cmat.shape[0]
    output=0
    for i in range(dim):
        for j in range(dim):
            if j<=i:
                continue
            #print 'do (%d,%d)'%(i,j)
            output+=cmat[i,j]
    return output

input=sys.argv[1]
threshold=0.5
output_f='remap.txt'
if len(sys.argv)>2:
    threshold=float(sys.argv[2])
mode=1 # mode 1: group with less confused, mode 2: group with large confused
if len(sys.argv)>3:
    mode=int(sys.argv[3])
if len(sys.argv)>4:
    output_f=sys.argv[4]
rows=[]
with open(input,'rb') as f:
    for ln in f:
        line=ln.rstrip('\n')
        if line =='' or line is None:
            continue
        data=line.split(',')
        rows.append(data)
n=len(rows)
cmat=np.zeros((n,n),dtype=np.float)
for i in range(n):
    row=rows[i]
    for j in range(n):
        val=row[j]
        cmat[i,j]=val
        
rooms=[]
correl=[]
for i in range(n):
    row=cmat[i,:]
    gt=i
    createRoom=False
    if rooms==[] :
        rooms.append([])
        rooms[0].append([i,row])
    else:
        rc=0
        for room in rooms:
            nMember=len(room)
            localCMat=np.zeros((nMember+1,nMember+1),dtype=np.float)
            ordered_indices=np.zeros((nMember+1,nMember+1),dtype=np.int)
            for r in range(nMember):
                cls=room[r][0]
                ordered_indices[r,r]=cls
                for dr in range(nMember+1):
                    ordered_indices[dr,r]=cls
            for r in range(nMember+1):
                ordered_indices[r,nMember]=gt
            for r in range(nMember):
                for dr in range(nMember+1):
                    localCMat[r,dr]=room[r][1][ordered_indices[r,dr]]
            for r in range(nMember+1):
                localCMat[nMember,r]=row[ordered_indices[nMember,r]]
            confused=compute_upperTriangle_sum(localCMat)
            if mode==1:
                if confused>threshold:
                    createRoom=True
                else:
                    rooms[rc].append([i,row])
                    createRoom=False
                    break
            else:
                if confused<threshold:
                    createRoom=True
                else:
                    rooms[rc].append([i,row])
                    createRoom=False
                    break
            rc+=1
            #print localCMat
            #print confused
            #print ''
    if createRoom:
        rooms.append([])
        rooms[len(rooms)-1].append([i,row])
print rooms
print len(rooms)
with open(output_f,'wb') as f:
    grp_map=np.zeros(n,dtype=int)
    for r in range(len(rooms)):
        ids=[]
        for rec in rooms[r]:
            ids.append(rec[0])
            grp_map[rec[0]]=r
        print ids
        print ''
    for idx in grp_map:
        f.write('%d\n'%idx)

print 'finished'