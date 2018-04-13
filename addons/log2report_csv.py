import os,sys
import numpy as np
import csv

input_file=sys.argv[1]
output_file=sys.argv[2]
header='I0315'
if len(sys.argv)>3:
    header=sys.argv[3]

keys=[]
epoch_data=[]
epoch_row=[]
with open(input_file,'r') as f:
    for ln in f:
        line=ln.rstrip('\n')
        if line.startswith(header) and 'Test net output' in line:
            fields=line.split(':')
            spaces=fields[len(fields)-2].split(' ')
            num=int(spaces[len(spaces)-1][1:])
            attr=fields[len(fields)-1]
            if num==0 and epoch_row!=[]:
                epoch_data.append(epoch_row)
                epoch_row=[]
            key=''
            if 'loss' in attr:
                key=attr.split('(')[0].split('=')[0]
                loss=float(attr.split('(')[0].split('=')[1])
                epoch_row.append(loss)
            else:
                key=attr.split('=')[0]
                epoch_row.append(float(attr.split('=')[1]))
            if len(epoch_data)==0 or epoch_data==[]:
                keys.append(key)
print keys

with open(output_file,'wb')as f:
    csv_writer=csv.writer(f,delimiter=',')
    csv_writer.writerow(keys)
    csv_writer.writerows(epoch_data)
print 'finished'