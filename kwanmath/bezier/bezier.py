import numpy as np

def interp(P0,P1,t):
    return (1-t)*P0+t*P1

def deCasteljau(P:np.array,t:float)->np.array:
    n=P.shape[1]
    if n==1:
        return P
    else:
        nP=np.zeros((P.shape[0],n-1))
        for i in range(n-1):
            nP[:,i]=interp(P[:,i],P[:,i+1],t)
        return deCasteljau(nP,t)

def B(i:int,t:float)->float:
    if i==0:
        return (1-t)**3
    elif i==1:
        return 3*(1-t)**2*t
    elif i==2:
        return 3*(1-t)*t**2
    elif i==3:
        return t**3

def bezier(P:np.array,t:float)->np.array:
    return (P[:,0]*B(0,t)+
            P[:,1]*B(1,t)+
            P[:,2]*B(2,t)+
            P[:,3]*B(3,t))

def flatness(P:np.array)->float:
    """
    Calculate the flatness of a given curve
    From https://www.joshondesign.com/2018/07/11/bezier-curves and
    https://hcklbrrfnn.wordpress.com/2012/08/20/piecewise-linear-approximation-of-bezier-curves/

    """
    u=(3*P[:,1]-2*P[:,0]-P[:,3])**2
    v=(3*P[:,2]-2*P[:,3]-P[:,0])**2
    w=np.where(u<v)
    u[w]=v[w]
    return np.sum(u)

def split(P:np.array,t:float)->tuple:
    p01=interp(P[:,0],P[:,1],t)
    p12=interp(P[:,1],P[:,2],t)
    p23=interp(P[:,2],P[:,3],t)
    p012=interp(p01,p12,t)
    p123=interp(p12,p23,t)
    p0123=interp(p012,p123,t)
    return (np.hstack((P[:,0],p01,p012,p0123)),np.hstack((p0123,p123,p23,P[:,1])))

def flatten(P:np.array,tol:float=1)->np.array:
    """
    Return a polyline which approximates the given cubic Bezier curve to within
    the given tolerance
    """
    if flatness(P)<tol:
        return np.hstack((P[:,0],P[:,3]))
    else:
        P0,P1=split(P,0.5)
        return np.hstack((flatten(P0,tol),flatten(P1,tol)))

def arc_l90(theta:float)->np.array:
    """
    Create a cubic Bezier curve which approximates a circular arc. Based on
    https://pomax.github.io/bezierinfo/#circles_cubic .

    :param theta: Angle of arc. The arc start on the +X axis and grows towards +Y. The math works up to
    180deg, but the approximation is noticeably worse, so we recommend only
    using this up to 90deg. At this angle, the maximum error between an actual circle and this curve
    is 2.7e-4 (2.7 unit error on a circle with radius 10,000 units).
    :return: Control points for a single Bezier curve which approximates the arc, in the form of a 2-row, 4-column
             numpy array.
    """
    k=4*np.tan(theta/4)/3
    c=np.cos(theta)
    s=np.sin(theta)
    result=np.zeros((2,4))
    result[:,0]=(1,0)
    result[:,1]=(1,k)
    result[:,2]=(c+k*s,s-k*c)
    result[:,3]=(c,s)
    return result

