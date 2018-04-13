class Package(object):
    def __init__(self):
        self.c=0
        
    def getC(self):
        return self.c

def add(a,b,res):
    res[0]=a*(1+b)
    
a=2
b=2
arr=[]
data=Package()
arr.append(data.c)
add(a,b,arr)
print data.c