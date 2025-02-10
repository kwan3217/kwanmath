"""
This script contains functions for fitting, evaluating, and identifying properties of conic sections in 2D space.

Functions:
- fit_conic: Fits a conic section to a given set of 2D points and returns the coefficients.
- eval1_conic: Evaluates the conic section using the two-branch method to get x and y coordinates for plotting.
- identify_conic: Identifies the properties (center, semimajor axis, semiminor axis) of the conic section.
- eval2_conic: Evaluates the conic section and returns points for plotting.

Created: 2/5/25
"""
import numpy as np


def fit_conic(v:np.ndarray,scale:float=1.0)->tuple[float,float,float,float,float,float]:
    """
    Calculate the coefficients of the quadratic in two variables that
    best fits the data values. This quadratic is:

      A'*x**2+B'*x*y+C'*y**2+D'*x+E'*y+F'=0

    These coefficients are not all independent -- they can all be multiplied
    by the same value and the same (x,y) pairs will still satisfy the equation.
    Therefore we will divide by -F' and get the equation:

      A*x**2+B*x*y+C*y**2+D*x+E*y-1=0
      A*x**2+B*x*y+C*y**2+D*x+E*y=1

    We will never speak of F, nor A' through E' again. A through E carry exactly
    the same information and are perfectly usable. If you need an F, remember that
    it is -1.

    :param v: 2xN vector of points in 2D space, N>=5
    :param scale: Scaling factor, to help with numerical
                  issues. The returned coefficients are NOT
                  inverse scaled -- you should pass the
                  same scale to other functions in this library
    :return: Coefficients A,B,C,D,E of the conic
    """
    # We are going to use the numpy library, specifically np.linalg.lstsq()
    # to solve this. It finds the value of column vector x which "solves"
    # A @ x = b. A is the design matrix -- it has one column for each basis
    # function, one row for each independent variable value, and is the
    # value of the basis function at the point. The independent variable
    # may be a vector, and in this case it is.
    #
    # A[i,j]=f_j(v[:,i])
    #
    # Each row of the design matrix is dotted with the coefficients in a
    # column vector x and is supposed to equal the element in the
    # corresponding column vector b. If the equation has an equal number
    # of basis functions and independent variable samples, then we usually
    # get an exact solution. If there are more independent variable
    # samples, the problem is overdetermined and the solver gives us the
    # one x that minimizes the sum of (b[i]-A[i,:]@x)**2 over all data
    # points.
    #
    # Our conic section problem can in fact be set up this way.
    # In our case we have f_0(v)=x**2 and therefore x[0]=A
    #                     f_1(v)=x*y  and therefore x[1]=B
    #                     f_2(v)=y**2 and therefore x[2]=C
    #                     f_3(v)=x    and therefore x[3]=D
    #                     f_4(v)=y    and therefore x[4]=E
    #
    # The vector x is unknown and is what we will solve for. The right hand side
    # vector b is a column vector where each element is the value of the right
    # hand side that we want for one of the independent vectors. Well in our case
    # we want the right hand side to always be 1 so we want b to be a column
    # vector N elements high, all 1. We _could_ solve it with the normal equations
    #
    #  A^T @ A @ a= A^T @ b
    #
    # which is solved by finding (A^T@A)^-1. We then have:
    #
    # (A^T@A)^-1@A^T@A@a=(A^T@A)^-1@A^T@b
    #                  a=(A^T@A)^-1@A^T@B
    # Instead we use np.linalg.lstsq() to do the solution in a way that's
    # supposed to have better numerical stability.
    xp=v[0,:]/scale
    yp=v[1,:]/scale
    A = np.column_stack((xp ** 2, xp * yp, yp ** 2, xp, yp))
    b = np.ones_like(xp).reshape(-1, 1)
    (Ap,Bp,Cp,Dp,Ep),resid,rank,s=np.linalg.lstsq(A,b)
    return Ap,Bp,Cp,Dp,Ep


def eval1_conic(Ap:float,Bp:float,Cp:float,Dp:float,Ep:float,scale:float=1.0)->tuple[np.array,np.array,np.array]:
    """
    Evaluate the given conic using the two-branch method
    :param Ap: Coefficient of xp**2
    :param Bp: Coefficient of xp*yp
    :param Cp: coefficient of yp**2
    :param Dp: coefficient of xp
    :param Ep: coefficient of yp
    :param scale: Scale factor to apply to return results, should equal that passed to fit_conic
    :return: A tuple:
      * x variable
      * y0 variable for one half of the ellipse. Plot it with plt.plot(x,y0).
      * y1 variable for the other half of the ellpise. Plot it with plt.plot(x,y1).
    """
    # Now we have Ap*xp**2+Bp*xp*yp+Cp*yp**2+Dp*xp+Ep*yp-1=0, which can be solved for y given x
    # since it's quadratic in y. We have
    # ```
    # from sympy import *
    # A,B,C,D,E,x,y=symbols('A B C D E x y')
    # conic=A*x**2+B*x*y+C*y**2+D*x+E*y
    # a1=collect(conic,y).coeff(y,2)
    # b1=collect(conic,y).coeff(y,1)
    # c1=collect(conic,y).coeff(y,0)
    # print("a1=",a1)
    # print("b1=",b1)
    # print("c1=",c1)
    # a1_sym,b1_sym,c1_sym=symbols('a1 b1 c1')
    # conicy=a1_sym*y**2+b1_sym*y+c1_sym
    # y0,y1=solve(Eq(conicy,0),y)
    # print("y0=",y0)
    # print("y1=",y1)
    # ```
    # a1 = C
    # b1 = B * x + E
    # c1 = A * x ** 2 + D * x
    # y0 = (-b1 - np.sqrt(-4 * a1 * c1 + b1 ** 2)) / (2 * a1)
    # y1 = (-b1 + np.sqrt(-4 * a1 * c1 + b1 ** 2)) / (2 * a1)
    # There is one solution at the left and right end of the ellipse, and two solutions
    # in between, like is normal for the quadratic. The extremes in x then are found by
    # finding when the discriminant exactly equals zero, and the discriminant is quadratic
    # in x:
    # ```
    # delta=expand(b1**2-4*a1*c1)
    # print("delta=",delta)
    # a2=collect(delta,x).coeff(x,2)
    # b2=collect(delta,x).coeff(x,1)
    # c2=collect(delta,x).coeff(x,0)
    # print("a2=",a2)
    # print("b2=",b2)
    # print("c2=",c2)
    # ```
    # delta= -4*A*C*x**2 + B**2*x**2 + 2*B*E*x - 4*C*D*x + E**2
    a2 = -4 * Ap * Cp + Bp ** 2
    b2 = 2 * Bp * Ep - 4 * Cp * Dp
    c2 = 4 * Cp + Ep ** 2
    # Now we set delta to 0 and solve for x using the quadratic formula again
    # ```
    # a2_sym,b2_sym,c2_sym=symbols('a2 b2 c2')
    # delta_sym=a2_sym*x**2+b2_sym*x+c2_sym
    # x0,x1=solve(Eq(delta_sym,0),x)
    # print("x0=",x0)
    # print("x1=",x1)
    # ```
    xp0 = (-b2 - np.sqrt(-4 * a2 * c2 + b2 ** 2)) / (2 * a2)
    xp1 = (-b2 + np.sqrt(-4 * a2 * c2 + b2 ** 2)) / (2 * a2)
    if xp1 < xp0:
        xpt = xp0
        xp0 = xp1
        xp1 = xpt
    xp = np.arange(xp0, xp1, 1 / scale)
    a1 = Cp
    b1 = Bp * xp + Ep
    c1 = Ap * xp ** 2 + Dp * xp - 1
    #Sometimes discriminant is slightly less than 0. First observed time it was -8e-16.
    disc=b1**2-4*a1*c1
    disc[np.logical_and(disc<0.0,np.isclose(disc,0.0))]=0.0
    yp0 = (-b1 - np.sqrt(disc)) / (2 * a1)
    yp1 = (-b1 + np.sqrt(disc)) / (2 * a1)
    x=xp*scale
    y0=yp0*scale
    y1=yp0*scale
    return x,y0,y1


def identify_conic(Ap:float, Bp:float, Cp:float, Dp:float, Ep:float, scale:float=1.0)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Identify the properties of a conic described by the given coefficients
    :param Ap: Coefficient of xp**2
    :param Bp: Coefficient of xp*yp
    :param Cp: coefficient of yp**2
    :param Dp: coefficient of xp
    :param Ep: coefficient of yp
    :param scale: Scale factor to apply to return results, should equal that passed to fit_conic
    :return: Tuple of 2-element column vectors, all with scaling applied to be in original input units:
      * center of ellipse
      * Direction and length of semimajor axis
      * Direction and length of semiminor axis
    The code carefully figures out which axis is longer and returns the longer one before the shorter one.
    With this you can do things like:
    ```
    cv,av,bv
    M=np.column_stack((av,bv))
    q=np.arange(0,2*np.pi,0.01)
    c=np.cos(q)
    s=np.sin(q)
    ru=np.row_stack((c,s))
    re=M@ru+cv
    ```
    Based on algorithm from https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections#Central_conics
    as of 2025-02-05
    In particular we use the "alternative approach" to finding the center, and the eigenvector and eigenvalue
    method for getting the axes.

    This code might handle multiple ellipses if the coefficients properly broadcast,
    but it hasn't been tested for this yet.
    """
    Fp=-1.0 # Doesn't scale since it doesn't multiply either x or y
    # Create the two matrices. This one is the _matrix of the quadratic equation_...
    Aqp=np.row_stack((np.column_stack((Ap,Bp/2,Dp/2)),
                      np.column_stack((Bp/2,Cp,Ep/2)),
                      np.column_stack((Dp/2,Ep/2,Fp))))
    # and this one is the _matrix of the quadratic form_ IE only dealing with terms
    # of power exactly 2.
    A33p=Aqp[0:2,0:2]
    # Figure the center point by the "alternative approach"
    cvp=np.linalg.inv(A33p) @ np.row_stack((-Dp/2,
                                            -Ep/2))
    # Figure the K constant, the ratio of the determinants
    Kp=-np.linalg.det(Aqp)/np.linalg.det(A33p)
    # Find the eigenvectors and eigenvalues, being careful
    # how to dissect the eigenvectors out of the return result
    (lam1p,lam2p),v=np.linalg.eig(A33p)
    v1=v[:,0]
    v2=v[:,1]
    #xpp**2*lam1p+ypp**2*lam2p=Kp
    #xpp**2*lam1p/Kp+ypp**2*lam2p/Kp=1
    #xpp**2*1/ap**2+ypp**2*1/bp**2=1
    #1/ap**2=lam1p/Kp
    #ap**2=Kp/lam1p
    #ap=sqrt(Kp/lam1p)
    #1/bp**2=lam2p/Kp
    #bp**2=Kp/lam2p
    #bp=sqrt(Kp/lam2p)
    # Figure the axis length that goes with each eigenvector
    ap=np.sqrt(Kp/lam1p)
    bp=np.sqrt(Kp/lam2p)
    # Multiply by the (unit) vector direction to get the axis vector
    avp=ap*v1
    bvp=bp*v2
    # For those cases where the a axis is shorter than the b axis, swap them
    # so a is always the major axis.
    switch=ap<bp
    t=avp[:,switch]
    avp[:,switch]=bvp[:,switch]
    bvp[:,switch]=t
    # Correct the vectors for scale and return
    cv=cvp*scale
    av=avp*scale
    bv=bvp*scale
    return cv,av,bv


def eval2_conic(Ap:float,Bp:float,Cp:float,Dp:float,Ep:float,scale:float=1.0)->np.array:
    """
    :param Ap: Coefficient of xp**2
    :param Bp: Coefficient of xp*yp
    :param Cp: coefficient of yp**2
    :param Dp: coefficient of xp
    :param Ep: coefficient of yp
    :param scale: Scale factor to apply to return results, should equal that passed to fit_conic
    :return: stack of 2D vectors of shape 2xN, where N is the number of points it's evaluated at,
             about 200*pi.
    """
    q=np.arange(0,np.pi*2,0.01)
    c=np.cos(q)
    s=np.sin(q)
    v=np.row_stack((c,s))
    cv,av,bv=identify_conic(Ap, Bp, Cp, Dp, Ep, scale)
    M=np.column_stack((av,bv))
    result=M@v+cv
    return result


def vector_to_conic(*,cv:np.ndarray,av:np.ndarray,bv:np.ndarray,scale:float=1.0,get_points:bool=False):
    """
    Calculate the A-E coefficients given the center and principal axis vectors
    :param cv: Center of ellipse
    :param av: Vector with length equal to semimajor axis and direction of that principal axis
    :param bv: Vector with length equal to semiminor axis and direction of that principal axis
    :param scale: Pass along to fit_conic
    :param get_points: If True, return the points we generate too as another element of the tuple
    :return: A tuple:
      Ap,Bp,Cp,Dp,Ep scaled conic coefficients
      points: 2x5 numpy array of column vectors of 5 points on the ellipse
    """
    q = np.arange(5) / 5 * 2 * np.pi
    c = np.cos(q)
    s = np.sin(q)
    vu = np.row_stack((c, s))
    M = np.column_stack((av, bv))
    ve = M @ vu + cv.reshape(-1, 1)
    Ap, Bp, Cp, Dp, Ep = fit_conic(ve, scale=scale)
    result=(Ap,Bp,Cp,Dp,Ep)
    if get_points:
        result=result+(ve,)
    return result





