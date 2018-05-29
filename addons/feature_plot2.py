import sys,os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
from sklearn import svm,datasets
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.font_manager import FontProperties

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy,labels, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    #Z=clf
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    print np.unique(Z)
    if labels!=[]:
        fmt={}
        fmt_diff={}
        out_lv=[]
        for l in out.levels:
            for li in range(len(labels)):
                ival=int(labels[li])
                diff=abs(l-ival)
                if diff<2:
                    if l in fmt_diff.keys():
                        if fmt_diff[l]>diff:
                            fmt_diff[l]=diff
                            fmt[l]=labels[li]
                    else:
                        fmt_diff[l]=diff
                        fmt[l]=labels[li]
                        out_lv.append(l)
                else:
                    if l not in fmt_diff.keys():
                        fmt[l]=''
        print fmt
        print out_lv
        ax.clabel(out,out_lv,inline=True,fmt=fmt,fontsize=10)
    return out


def index2rgb(index,nlen):
    nc=65536
    color=255+int(np.floor((float(index)*nc/nlen)))
    r=(color>>16)&int('0xFF',16)
    g=(color>>8)&int('0xFF',16)
    b=color&int('0xFF',16)
    return [float(r)/255,float(g)/255,float(b)/255,1]

srcs=sys.argv[1].split(',')
boundary=None
super_lbl={}
output_clss=[]
if len(sys.argv)>2:
    output_clss=sys.argv[2].split(',')

cls={}
b_cls={}
total=0
clss=[]
fs=[]
features={}
ii=0
ndata=0
for src in srcs:
    print src
    if ii==0:
        with open(src,'r') as f:
            for ln in f:
                line=ln.rstrip('\n')
                fields=line.split(',')
                gt=int(float(fields[2]))
                if output_clss !=[] and str(gt) not in output_clss:
                    continue
                ndata+=1
                if gt not in cls.keys():
                    cls[gt]=None
                clss.append(gt)
        print ndata
    f_feature=np.zeros((ndata,2),np.float)
    with open(src,'r') as f:
        total=0
        for ln in f:
            line=ln.rstrip('\n')
            fields=line.split(',')
            gt=int(float(fields[2]))
            if output_clss!=[] and str(gt) not in output_clss:
                continue
            f_feature[total,0]=float(fields[0])
            f_feature[total,1]=float(fields[1])
            features[ii]=f_feature
            total+=1
    ii+=1

linked_cls=None
cc=0
            
max_cls=len(cls.keys())
for i in range(max_cls):
    #cls[cls.keys()[i]]=index2rgb(i, max_cls)
    cls[cls.keys()[i]]=i*255/max_cls
    #print cls[cls.keys()[i]]
feat2d=np.zeros((total,2),dtype=np.float32)
#data_rgb=np.zeros((total,3),dtype=np.int)
data_color=np.zeros((total,1),dtype=np.int)
y=np.zeros(total,dtype=np.int)
total=0
print 'preparing for dataset'
sgroup_feat={}
sgroup_color=[]
sgroup_count={}
sgroup_label=[]
cmap=matplotlib.cm.get_cmap('Paired')

for n in range(ndata):
    gt=clss[n]
    data_color[total,:]=cls[gt]
    if gt not in sgroup_label:
        sgroup_count[gt]=0
        sgroup_label.append(gt)
        sgroup_color.append(cmap(0.1+(float(gt)/18)))
    sgroup_count[gt]+=1
    y[total]=gt
    total+=1   

print 'ready for show'

fig=plt.figure(1)     

#clf=svm.LinearSVC(C=C)
#clf=clf.fit(feat2d,y)
#clf=KNeighborsClassifier(n_neighbors=20) # 100 for demonstrate learner acts
#clf.fit(feat2d,y)
#xx,yy=make_meshgrid(feat2d[:,0], feat2d[:,1],h=.5)
labels=[]
for icls in sgroup_label:
    labels.append(str(icls))

#plot_contours(plt, clf, xx, yy,[],cmap=plt.cm.Pastel1,s=20,edgecolors='k')
d=0

print len(srcs)
print len(features)
d=0
for key in sgroup_label:
    t_feat2d=np.zeros((sgroup_count[key]*len(srcs),2),dtype=np.float32)
    k=0
    for u in range(ndata):
        if y[u]!=key:
            continue
        for ch in range(len(srcs)):
            feat2d=features[ch]
            t_feat2d[k,:]=feat2d[u,:]
            if u==0:
                print feat2d[u,:]
            k+=1
    print t_feat2d.shape
    plt.scatter(t_feat2d[:,0], t_feat2d[:,1], s=5, color=sgroup_color[d],label='posture %d'%(key+1))
    d+=1

fp=FontProperties()
fp.set_size('small')
fig.subplots_adjust(top=0.9,bottom=0.2,left=0.1,right=0.8)
plt.legend(loc=9,bbox_to_anchor=(0.5, -0.05),prop=fp,ncol=len(sgroup_label)/2)
         
 
plt.show()

