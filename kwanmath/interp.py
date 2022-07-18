"""
Interpolation functions the Kwan Systems way
"""
from scipy.interpolate import interp1d
import numpy as np

def linterp(x0, y0, x1, y1, x, bound=False):
    """
    Linear interpolation

    :param x0: Input value at one end of line
    :param y0: Output value which x0 maps onto
    :param x1: Input value at other end of line
    :param y1: Output value which x1 maps onto
    :param x:  Input value(s)
    :param bound: If True, return the correct y endpoint value if x is outside the x endpoints
    :return:   Output value(s).

    Note: All and only operators +,-,*, and / are used. Any type which supports
          these may be used as parameters, including numpy arrays. In case arrays
          are used, all broadcasting is supported and inputs must be compatible
          by broadcasting.

          The usual case is for x0,y0,x1,y1 to all be scalar and for x to be
          either scalar or array, in which case the output will be the same size
          and shape as the input x.

          If bound= is set, this uses < and > and therefore only works for types which can
          be compared, so scalars, not numpy arrays.
    """
    if bound:
        if x<x0:
            return y0
        if x>x1:
            return y1
    t = (x - x0) / (x1 - x0)
    return (1 - t) * y0 + t * y1

def trap(x0,x1,x2,x3,y0,y1,x):
    if x<x0:
        return y0
    elif x<x1:
        return linterp(x0,y0,x1,y1,x)
    elif x<x2:
        return y1
    elif x<x3:
        return linterp(x2,y1,x3,y0,x)
    else:
        return y0

def smooth(s,p0,p1):
    """
    Do a boxcar average on a 1D dataset
    :param v:
    :return:
    """
    result=s*0
    count=0
    for p in range(p0,p1):
        result[-p0:-p1]+=s[-p0+p:len(s)-p1+p]
        count+=1
    result/=count
    result[:-p0]=s[:-p0]
    result[-p1:]=s[-p1:]
    return result

class tableterp:
    """
    Acts like a function that returns a function.
    """
    def __init__(self,x,y,smooth0=None,smooth1=None):
        self._x=x
        if smooth0 is None:
            self._y=y
        else:
            if len(y.shape)==1:
                self._y=smooth(y,smooth0,smooth1)
            elif len(y.shape)==2:
                self._y=np.zeros(y.shape)
                for i in range(y.shape[0]):
                    self._y[i,:]=smooth(y[i,:],smooth0,smooth1)
            elif len(y.shape) == 2:
                self._y=np.zeros(y.shape)
                for i in range(y.shape[1]):
                    for j in range(y.shape[2]):
                        self._y[:,i,j] = smooth(y[:,i,j],smooth0,smooth1)
        if len(self._y.shape)==2:
            self.axis = 1
        else:
            self.axis=0
        self._yi=interp1d(self._x,self._y,axis=self.axis,assume_sorted=True,copy=False)
    def __call__(self,x=None):
        if x is None:
            result=self._y.view()
            result.flags.writeable=False
            return result
        if len(self._y.shape)==2:
            return self._yi(x)[:,None]
        else:
            return self._yi(x)


