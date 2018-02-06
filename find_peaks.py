import peakutils as pk
import numpy as np
from matplotlib.mlab import find
from tomkin import detect_rpeak
from pylab import *
def find_rr_interval(data):
    ind = pk.indexes(data[:,2],thres=.7,min_dist=15)
    # figure()
    # plot(data[:,0],data[:,2],data[ind,0],data[ind,2],'*')
    # show()
    ts = data[ind,0]
    rr_int = np.diff(ts)
    ts = np.array(ts[1:])
    ind = np.where((rr_int>300) & (rr_int < 2000))[0]
    rr_int = rr_int[ind]
    ts = ts[ind]
    return ts,rr_int