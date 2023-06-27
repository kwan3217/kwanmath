"""
Stuff to do with Gaussian normal distributions and such
"""

import numpy as np
import scipy.optimize as opt
from collections import namedtuple
from kwanmath.vector import vlength

class bounded:
    """
    Function with the following properties:
      #Monotonically increasing
      #Unbounded domain -- x can be any real number from -inf to +inf
      #Bounded range -- result has an asymtote of y0 as x approaches -inf, and y1 as x approaches +inf
      #Slope of 1 at x=x0
    Used to allow the optimizer to pick any real value for a parameter,
    and map it one-to-one to a value which is within the given bounds.
    The slope=1 property makes the function act nicely if the parameter
    is in the neighborhood of x0.
    """
    def __init__(self,y0=0,y1=1,x0=0):
        self.y0=y0
        self.y1=y1
        self.x0=x0
    def __call__(self,x):
        a=4 / (self.y1 - self.y0)
        y=(self.y1-self.y0)/(1+np.exp(-a*(x-self.x0)))+self.y0
        return y
    def inverse(self,y):
        a=4 / (self.y1 - self.y0)
        x=-np.log((self.y1-self.y0)/(y-self.y0)-1)/a+self.x0
        return x
    def __repr__(self):
        return "bounded(y0=%f,y1=%f,x0=%f)"%(self.y0,self.y1,self.x0)


class positive:
    """
    Function with the following properties:
      #Monotonically increasing
      #Unbounded domain -- x can be any real number from -inf to +inf
      #Bounded range -- result has an asymtote of 0 as x approaches -inf, but no asymtote as x approaches +inf
      #Slope of 1 at x=x0
    """
    def __init__(self,x0=0):
        self.x0=x0
    def __call__(self,x):
        y=np.exp(x-self.x0)
        return y
    def inverse(self,y):
        x=np.log(y)+self.x0
        return x
    def __repr__(self):
        return "positive(x0=%f)"%(self.x0)


_bound_sig=positive()
_bound_rho=bounded(y0=-1,y1=1)


def twoD_Gaussian(x,y,amplitude,xc,yc,sigma_x,sigma_y,rho,offset):
    """

    :param x:
    :param y:
    :param amplitude:
    :param xc:
    :param yc:
    :param sigma_x:
    :param sigma_y:
    :param rho:
    :param offset:
    :return:
    """
    xc = float(xc)
    yc = float(yc)
    P00=sigma_x**2
    P11=sigma_y**2
    P01=sigma_x*sigma_y*rho
    P=np.array([[P00,P01],[P01,P11]])
    Pinv=np.linalg.inv(P)
    xv=np.stack([x-xc,y-yc],axis=2)
    xv=xv.reshape(xv.shape+(1,))
    xvT=np.transpose(xv,[0,1,3,2])
    xpo1=xvT @ Pinv
    xpo=xpo1 @ xv
    xpo=xpo.reshape(xv.shape[0:-2])
    g=offset + amplitude*np.exp(-xpo/2)
    return g


def twoD_Gaussian_opt(xy, amplitude, xc, yc, bsigma_x, bsigma_y, brho, offset):
    """
    Generate a 2D Gaussian image. Arguments are a bit weird as this has to be
    compatible with scipy.optimize.opt
    :param xy: tuple of pixel coordinates
    :param amplitude: amplitude of Gaussian. Value at center will be amplitude+offset
    :param xc: center of Gaussian in x direction
    :param yc: center of Gaussian in y direction
    :param bsigma_x: bounded standard deviation of Gaussian in x direction
    :param bsigma_y: bounded standard deviation of Gaussian in Y direction
    :param brho:     bounded correlation coefficient
    :param offset:   Floor of Gaussian
    :return: A 2D Gaussian but encoded in a 1D array so as to be compatible with scipy.optimize.opt
    """
    return twoD_Gaussian(xy[0],xy[1],amplitude,xc,yc,
                         _bound_sig(bsigma_x),_bound_sig(bsigma_y),_bound_rho(brho),offset).ravel()


def correlation_matrix(P:np.array)->np.array:
    """
    Convert a covariance matrix to a correlation matrix. This
    has the same information as covariance, but is a little easier
    for humans to interpret
    :param P: Covariance matrix. Each on-diagonal element is the
              variance sigma_i**2 of element i, and each off-diagonal
              element is the covariance sigma_i*sigma_j*rho_ij of
              element i and j
    :return: Correlation matrix. Each on-diagonal element is the
              standard deviation sigma_i of element i, and each off-diagonal
              element is the correlation coefficient rho_ij of
              element i and j
    """
    result=P*0
    for i in range(P.shape[0]):
        result[i,i]=np.sqrt(P[i,i])
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            if i!=j:
                result[i,j]=P[i,j]/result[i,i]/result[j,j]
    return result

def fit_twoD_Gaussian(img,dn_size:float=1.0):
    """
    Fit a 2D Gaussian surface to a given image
     
    :param img: 2D numpy array
    :return: Tuple:
      * Amplitude of Gaussian
      * horizontal coordinate of center
      * vertical coordinate of center 
      * sigma of distribution in horizontal direction
      * sigma of distribution in vertical direction
      * rho of distribtion
      * offset of Gaussian
      * 2D array of given Gaussian evaluated over the image
    """
    result=namedtuple("fit_twoD_Gaussian_result",
                      "amp xc yc sigx sigy rho ofs eval cov")
    data=img.ravel()
    x = np.linspace(0, img.shape[1]-1,img.shape[1])
    y = np.linspace(0, img.shape[0]-1,img.shape[0])
    x, y = np.meshgrid(x, y)
    g_offset=np.min(img)
    g_amplitude=np.max(img)-g_offset
    g_yc,g_xc=np.unravel_index(np.argmax(img,axis=None),img.shape)
    g_sigma_x=5
    g_sigma_y=5
    g_rho=0
    popt, pcov = opt.curve_fit(twoD_Gaussian_opt, (x,y), data,
                               p0=(g_amplitude,g_xc,g_yc,
                                   _bound_sig.inverse(g_sigma_x),
                                   _bound_sig.inverse(g_sigma_y),
                                   _bound_rho.inverse(g_rho),g_offset),
                               sigma=dn_size*np.ones(x.shape[0]*x.shape[1]),
                               absolute_sigma=True)
    data_fitted = twoD_Gaussian_opt((x, y), *popt).reshape(img.shape)
    popt[3]=_bound_sig(popt[3])
    popt[4]=_bound_sig(popt[4])
    popt[5]=_bound_rho(popt[5])
    result=result(amp=popt[0],xc=popt[1],yc=popt[2],sigx=popt[3],sigy=popt[4],rho=popt[5],ofs=popt[6],
                  eval=data_fitted,cov=pcov)
    return result


def mean_w(data:np.array,weight:np.array)->np.array:
    """
    :param data: A set of M, Nd column vector values, in the form of an NxM 2D array
    :param weight: 1D array of weights. Effectively a vector with weight x has
      exactly the same effect as x identical vectors of weight 1.0
    :return: mean of weighted dataset, in the form of an Nx1 column vector
    """
    xbar=np.sum(weight*data,axis=1)/np.sum(weight)
    return xbar


def std_w(data:np.array,weight:np.array)->np.array:
    """
    population standard deviation (ddof=0) of a weighted set of vectors
    :param data: A set of M, Nd column vector values, in the form of an NxM 2D array
    :param weight: 1D array of weights. Effectively a vector with weight x has
      exactly the same effect as x identical vectors of weight 1.0
    :return: standard deviation of weighted dataset, in the form of an Nx1 column vector
    """
    xbarw=mean_w(data,weight).reshape(-1,1)
    s=np.sqrt((np.sum(weight*(data-xbarw)**2,axis=1)/np.sum(weight)).reshape(-1,1))
    return s


def infamily(calcs:np.array,obss:np.array=None,nsig:float=5.0,weights:np.array=None)->np.array:
    """
    Calculate which members in a vector dataset are in and out of family

    :param obss: M observed Nd vector values, in the form of an NxM 2D array
    :param calcs: calculated vector values. If not present, it's equivalent to
                  treating all calcs as zero, IE treating obss as already the
                  deviation from expected
    :param nsig: Number of sigma off a value has to be in order to be considered out-of-family. The
      out-of-family measurement is taken as the weighted stdev of all the *other* O-C vectors, not
      including the vector currently under consideration
    :param weights: 1D array of weights. weighted standard deviation of a family is sqrt(sum(((O-C)*weight)**2)/sum(weight))
    :return: 1D boolean array, true for vectors which are in family, false for those that aren't.

    Known limitations:
    All of these can be fixed, but won't be until we demonstrate a need to do so
    * Data can't have any nonfinite values in it -- doing so will render the entire result nonfinite
    * Data must be a stack of vectors in the above form
    * Data is presumed to not edit itself to oblivion

    Example:
      observed_vectors,sig=...   #observed image coordinates and uncertainty based on star-finding from an image
      calculated_vectors=... #corresponding projected vectors stars based on pointing model parameter fit
      weight=1/sigma
      keep=infamily(calculated_vectors,observed_vectors,weight=weight,nsig=5)
      kept_calc_vectors=calculated_vectors[:,keep]

    """
    if obss is None:
        obss=calcs*0.0
    omc=obss-calcs
    lengths=vlength(omc)
    if weights is None:
        weights=lengths*0.0+1.0
    any_edited=True
    result=np.array([True]*len(weights))
    while any_edited:
        worst_dev=0.0
        worst_i=None
        any_edited=False
        for i in range(len(weights)):
            if result[i]:
                this_result=result.copy()
                this_result[i]=False
                xbar = mean_w(omc[:,this_result], weights[this_result])
                vec_onesig=std_w(omc[:,this_result], weights[this_result])
                onesig = vlength(vec_onesig)
                this_dev=vlength(omc[:,i]-xbar)
                if (this_dev/onesig)>worst_dev:
                    worst_dev=this_dev/onesig
                    worst_i=i
        if worst_dev>nsig:
            result[worst_i]=False
            any_edited=True
    return result


