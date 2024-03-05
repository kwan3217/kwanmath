"""
Wrapper around scipy.optimize.curve_fit
"""

from typing import Callable,Any,Optional, Iterable
from functools import partial

import numpy as np
import numpy.typing as npt
import scipy.optimize

class Constraint:
    pass

class bounded(Constraint):
    """
    Function with the following properties:
      #Monotonically increasing
      #Unbounded domain -- x can be any real number from -inf to +inf
      #Bounded range -- result has an asymtote of y0 as x approaches -inf, and y1 as x approaches +inf
      #Slope of 1 at x=x0=y0
    Used to allow the optimizer to pick any real value for a parameter,
    and map it one-to-one to a value which is within the given bounds.
    The slope=1 property makes the function act nicely if the parameter
    is in the neighborhood of x0.
    """
    def __init__(self,y0=0.0,y1=1.0,yg=None):
        """

        :param y0: lower bound
        :param y1: upper bound
        :param yc: point at which we want the slope to be 1.0

        See jupyter notebook "Nonlinear fitting constraints" for derivation
        """
        self.y0=y0
        self.y1=y1
        if yg is None:
            yg=(y1-y0)/2.0
        self.yg=yg
        eaxg=(y0-yg)/(yg-y1)
        self.a=(1+eaxg)**2/((y1-y0)*eaxg)
        self.xg=np.log(eaxg)/self.a
    def __call__(self,x):
        y=(self.y1-self.y0)/(1+np.exp(-self.a*(x-self.yg+self.xg)))+self.y0
        return y
    def inverse(self,y):
        x = (self.a * self.xg - self.a * self.yg + np.log((self.y1 - self.y0) / (y - self.y0) - 1)) / -self.a
        return x
    def __repr__(self):
        return "bounded(y0=%f,y1=%f,yg=%f)"%(self.y0,self.y1,self.yg)
    
    
class rbounded(bounded):
    def __init__(self,ry0=-1.0,ry1=1.0,yg=0.0):
        super().__init__(y0=yg+ry0,y1=yg+ry1,yg=yg)


class positive(Constraint):
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


def f_curve_fit(xdata,*int_params,f_int=None,pconst=None,vary=None,**kwargs):
    """
    Reshuffle parameters and arguments and call the wrapped cost function
    :param int_params: Internal parameters
    :param f:
    :param vary:
    :param int_args: Iterable of internal arguments. First n of these will be the
       frozen parameter values, remainder will be actual arguments passed to .minimize(args=(...))
    :return:
    """
    params=[]
    args=[]
    i_params=0
    i_args=0
    for i_vary,v in enumerate(vary):
        if v:
            try: #If v is a parameter transformation
                this_param=v(int_params[i_params])
            except:
                this_param=int_params[i_params]
            params.append(this_param)
            i_params+=1
        else:
            params.append(int_args[i_args])
            i_args+=1
    ydata=f_int(xdata,*params,**kwargs)
    return ydata


def curve_fit(f:Callable,
              xdata:npt.ArrayLike,
              ydata:npt.ArrayLike,
              p0:npt.ArrayLike,
              vary:Iterable[bool|Constraint],
              *,
              f_args:Optional[npt.ArrayLike]=None,
              f_kwargs:Optional[dict[str,Any]]=None,
              **kwargs)->npt.ArrayLike:
    """
    Wrapper around scipy.optimize.curve_fit. Given the following model:

      ydata=f(xdata,*params)+eps,

    figure the values of params that does a weighted minimization of eps. The wrapper
    is there to allow parameters to be easily activated or deactivated without changing
    the function f.

    Note that xdata and ydata do not need to be the same shape.

    All parameters are the same as scipy.optimize.curve_fit except for bounds

    :param f: function to evaluate, of the form f(xdata,*params,*f_args,**f_kwargs)
    :param xdata: independent values of function
    :param ydata: dependent values of function to match
    :param p0: Iterable of initial guesses to parameters to use to fit the curve
    :param f_args: Iterable of other data that the cost function needs, but are not to be optimized
    :param f_kwargs: Iterable of other data that the cost function needs, but are not to be optimized
    :param vary: Iterable of booleans, should be same length as p0. Each element is
       a boolean True or Constraint object for parameters which are allowed to be varied
       by the optimizer, or false for parameters to be held at their initial guesses.
    :return: tuple of:
      * array of parameter values which mimimizes the cost function. Any parameter where
        corresponding vary element is False will have exactly its initial guess value.
      * 2D array of covariance. For
    """
    #Move parameters to argument list as needed
    internal_p0=[]
    internal_pconst=[]
    for i_vary,v in enumerate(vary):
        if v:
            try:
                this_param=v.inverse(p0[i_vary])
            except:
                this_param=p0[i_vary]
            internal_p0.append(this_param)
        else:
            internal_pconst.append(p0[i_vary])
    ff=partial(f_curve_fit,f_int=f,pconst=internal_pconst,vary=vary,**f_kwargs)
    int_popt,int_pcov=scipy.optimize.curve_fit(ff,xdata.ravel(),ydata.ravel(),p0=internal_p0,**kwargs)
    i_int=0
    ext_popt=np.zeros(len(vary))
    ext_pcov=np.zeros((len(vary),len(vary)))
    for i_ext,v in enumerate(vary):
        if v:
            #Grab the computed result, and bump the result pointer
            try:
                ext_popt[i_ext]=v(int_popt[i_int])
            except:
                ext_popt[i_ext]=int_popt[i_int]
            ext_pcov[i_ext,i_ext]=int_pcov[i_int,i_int]
            i_int+=1
        else:
            #Grab the original parameter value and don't bump the result pointer
            ext_popt[i_ext]=params[i_ext]
            ext_pcov[i_ext,i_ext]=np.nan
    return ext_popt,ext_pcov