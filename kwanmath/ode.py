"""
Ordinary Differential Equation solvers. This solves first-order vector ordinary differential equations, and
any order vector equation can be reduced to first-order by adding more equations and state vector elements.

Differential equations are themselves expressed as Python functions which calculate the derivative
of a state given the state. They are of the form F(t,y) where:
   * t is the current value of the independent variable (time), which will always be a scalar. Many physical laws
       expressed as differential equations *don't* depend on time, only on various elements of the current state.
       You can do this by accepting but just not using t to calculate your derivatives. The function is allowed
       to use this to calculate derivatives which are functions of time as well as state (for instance throttle command
       of a rocket)
   * y is the current value of the dependent variable (state vector). This is mostly just subject to numpy broadcasting
       rules, allowing an interpretation from the following (possibly incomplete) list.
       - scalar, meaning the problem has a single-element state vector
       - stack of scalars, a 1D numpy array where each element represents an independent state,
       - row vector, a 1D numpy array which together represents a single state
       - column vector, a 2D Nx1 numpy array which represents a single state vector with N elements
       - Stack of column vectors, a 2D NxM numpy array which represents M different state vectors, each with N elements
       - Multi-dimensional stack of column vectors, a kD ...xNxM numpy array which represents a ...xM stack of
         independent N-element state vectors
       - Matrix, a 2D NxM numpy array which represents a single state matrix with NxM elements
       - Stack of matrices, a 3D LxNxM numpy array which represents a stack of L state matrices, each with size NxM
       - Multi-dimensional stack of matrices, a kD ...xNxM numpy array which represents a ... stack of independent
         NxM state matrices
       The integrator doesn't care about the interpretation, but the caller and differential equation should agree.

The return value dy/dt must be a result with the same dimensionality as y, where each element in the result is the
derivative of the corresponding element of the dependent variable with respect to the independent variable.

We use t for the independent variable and y for the dependent variable, because x is ambiguous. Often in dynamics, we
use x for the state vector which makes it dependent. Conversely when playing around with simple 1D problems, we use
x as the independent variable and y as dependent, since this is how we plot it.

The equation code should treat all of its arguments as input-only, because it is difficult to say what effect
changing an argument will have on other substeps.

You can use lambda functions as adapters to any existing function which calculates the derivative. For instance,
say you have a gravity function Fgrav where mu is a parameter that is constant over an integration, and there is no t
argument, since there is no time dependence.

def Fgrav(x, mu):
    x, y, dx, dy = x
    r = sqrt(x ** 2 + y ** 2)
    ddx = mu * x / r ** 3
    ddy = mu * y / r ** 3
    return np.array([dx, dy, ddx, ddy])

Instead of using Fgrav directly, use the following:

lambda t,y:Fgrav(x,mu=47)

You can use any callable that takes the right two parameters.

All solvers have the general form solver(F,x0,t0=0,n_step=None,t1=None,dt=None...) with the following parameters:

*F -- First-order vector differential equation
*t0 - initial value of t. The integrator itself doesn't use this for its own calculations, but it does use it to
      initialize the value of t it passes to the differential equation.
*y0 - initial state. As noted above, this can be a scalar, vector, matrix, or stack of independent runs of any of these.
      For instance, you can do a point cloud by passing M different column vectors of size N in a 2D NxM array.
*n_step - Number of steps to take. Notice that the derivative function may be evaluated more times than this --
      for instance fourth-order Runge-Kutta will evaluate four times in a step, at the beginning, middle (twice)
      and end of the step.
*t1 - final time to integrate to. The final step will end with its t>=t1 -- if t happens to exactly (floating-point
      precision) equal t1, it won't take another step, but if it is even 1 bit less, it will take one more step, putting
      t almost a full step beyond t1.
*dt - size of step to take. Intended to be "infinitesimal" which in a practical sense means as small as possible
      while considering run-time and round-off error.
Note that n_step, t1, and dt are overdetermined -- any two can be used to calculate the third. If all three are passed,
an implementation-defined one will be ignored and recalculated from the other two -- usually dt would be recalculated
from t1 such that the last step will exactly hit t1.

The return value is a tuple of the actual time and state at the end of the last step, and the actual time is guaranteed
to be >=t1. In many cases it will be exactly ==t1.

If you want a table of values, do it like this: The first row is t0,y0. Each subsequent row is
tn,yn=solver(t0=tnm1,y0=ynm1,t1=desired_tn,...). If you make lists, you can do it like this:

ts=[t0]
ys=[y0]
for desired_tn in ...:
    t,y=solver(t0=t[-1],y0=y[-1],t1=desired_tn)
    ts.append(t)
    ys.append(y)

This form is intended to allow adaptive step methods. In this case, you might pass an initial n_step and/or dt, but
also some error tolerance. The step size is adjusted to give the largest step which satisfies the tolerance.

"""
from typing import Union,Any
from collections.abc import Callable
import numpy as np

xtype=Union[float,np.array]
Ftype=Callable[[xtype,float,Any],xtype]

def calc_fixed_step(*,t0:float=0,n_step:int=None,t1:float=None,dt:float=None,fps:float=None)->tuple[int,float,float,float]:
    """
    Decorator which fills in the missing time step parameters

    :param t0: Initial time
    :param t1: Final time
    :param n_step: Number of time steps to take
    :param dt: Time between steps
    :param fps: Number of steps per time unit (frames per second)
    :return: Tuple of (t1,n_step,dt,fps). Each value is carefully handled
    such that the calling code should believe each result with all its heart,
    and not for instance try to calculate t1=t0+n_step*dt. This code is
    intended to do the best possible job in the face of floating-point
    precision and imperfectly representable numbers like 1/3 or 1/10.

    Note that if all four parameters are passed, the output is overdetermined and
    some of the inputs are ignored.

    fps is included because often the user will ask for each time unit to be
    chopped up into equal pieces, but the size of each piece is not perfectly
    representable as a floating point, IE 1/3 or 1/10. In this case, you pass
    fps=3 or fps=10. The code is then careful to use fps instead of dt to not
    scale up the error between 1/10 and float(1/10) 10 times.

    Calling code should always use returned t1 as the final step endpoint, even
    though it is not guaranteed (due to floating point precision) that
    t1==t0+n_step*dt. The actual last step might end at t0+n_step*dt, but
    it should be "close enough" to t1, and t1 is what the user intended to reach.
    This should help reduce round-off error in tables when repeatedly calling
    an integrator for each row.
    """
    try:
        if fps is not None:
            dt = 1 / fps
        if t1 is None:
            if fps is not None:
                t1 = t0 + n_step / fps
            else:
                t1 = t0 + n_step * dt
        elif n_step is None:
            if fps is not None:
                n_step = int(np.ceil((t1 - t0) * fps))
            else:
                n_step = int(np.ceil((t1 - t0) / dt))
        elif dt is None:
            dt = (t1 - t0) / n_step
        if fps is None:
            fps = 1 / dt
        return n_step,t1,dt,fps
    except Exception:
        raise ValueError(f"Time step is underdetermined -- t0={t0}, n_step={n_step}, t1={t1}, fps={fps}, dt={dt}")


def fixed_step(f:Callable)->Callable:
    """
    Decorator which fills in the missing time step parameters

    :param t0: Initial time
    :param t1: Final time
    :param n_step: Number of time steps to take
    :param dt: Time between steps
    :param fps: Number of steps per time unit (frames per second)
    :return: Tuple of (t1,n_step,dt,fps). Each value is carefully handled
    such that the calling code should believe each result with all its heart,
    and not for instance try to calculate t1=t0+n_step*dt. This code is
    intended to do the best possible job in the face of floating-point
    precision and imperfectly representable numbers like 1/3 or 1/10.

    Note that if all four parameters are passed, the output is overdetermined and
    some of the inputs are ignored.

    fps is included because often the user will ask for each time unit to be
    chopped up into equal pieces, but the size of each piece is not perfectly
    representable as a floating point, IE 1/3 or 1/10. In this case, you pass
    fps=3 or fps=10. The code is then careful to use fps instead of dt to not
    scale up the error between 1/10 and float(1/10) 10 times.

    Calling code should always use returned t1 as the final step endpoint, even
    though it is not guaranteed (due to floating point precision) that
    t1==t0+n_step*dt. The actual last step might end at t0+n_step*dt, but
    it should be "close enough" to t1, and t1 is what the user intended to reach.
    This should help reduce round-off error in tables when repeatedly calling
    an integrator for each row.
    """
    def inner(F:Ftype,x0:xtype,t0:float=0,n_step:int=None,t1:float=None,dt:float=None,fps:float=None)->tuple[float,xtype]:
        n_step,t1,dt,fps=calc_fixed_step(t0=t0,n_step=n_step,t1=t1,dt=dt,fps=fps)
        return f(F,x0,t0=t0,n_step=n_step,t1=t1,dt=dt,fps=fps)
    return inner

@fixed_step
def euler(F:Ftype,x0:xtype,t0:float=0,n_step:int=None,t1:float=None,dt:float=None,fps:float=None)->tuple[float,xtype]:
    """
    Take a fixed number of steps in a numerical integration of a differential
    equation using the Euler method.
    :param F: First-order vector differential equation as described above
    :param x0: Initial state
    :param t0: Initial time value
    :param nstep: Number of steps to take
    :param t1: final time value
    :param dt: Time step size
    :return: A tuple (t1,x1) of the time and state at the end of the final step. State x1
             will have same dimensionality as the input x0
    """
    x1=x0*1
    for i in range(n_step):
        x1+=dt*F(x=x1,t=t0+dt*i)
    return t1,x1

@fixed_step
def rk4(F:Ftype,x0:xtype,t0:float=0,n_step:int=None,t1:float=None,dt:float=None,fps:float=None)->tuple[float,xtype]:
    """
    Take a fixed number of steps in a numerical integration of a differential equation using the
    fourth-order Runge-Kutta method.
    :param F: First-order vector differential equation as described above
    :param x0: Initial state
    :param t0: Initial time value
    :param nstep: Number of steps to take
    :param t1: final time value
    :param dt: Time step size
    :return: A tuple (t1,x1) of the time and state at the end of the final step. State x1
             will have same dimensionality as the input x0
    """
    xp=x0*1
    for i in range(n_step):
        dx1=dt*F(xp      ,t0+dt*i     )
        dx2=dt*F(xp+dx1/2,t0+dt*i+dt/2)
        dx3=dt*F(xp+dx2/2,t0+dt*i+dt/2)
        dx4=dt*F(xp+dx3  ,t0+dt*i+dt  )
        xp=xp+(dx1+2*dx2+2*dx3+dx4)/6
    return t1,xp
