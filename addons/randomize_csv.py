import os,sys
import numpy as np
import pandas as pd

df=pd.read_csv(sys.argv[1],header=None)
ndf=df.sample(frac=1).reset_index(drop=True)
ndf.to_csv(sys.argv[2],index_label=False,index=False,header=False)
