from numbers import Number

import numpy as np

def vindex(v):
    if len(v.shape) > 2:
        return len(v.shape) - 2
    return 0

def vncomp(v):
    return v.shape[vindex(v)]

def vcomplimit(v, n):
    """
    Return a stack of vectors with the same shape as the input stack, but only
    including the first n vector components.

    :param v: input vector. Must have at least n components
    :param n: Number of vector components to keep
    :return:
    """
    if vindex(v) == 0:
        return v[:n, ...]
    else:
        return v[..., :n, :]

def vdot(a, b, array=False):
    """
    Dot product of two vectors or stack of vectors

    :param a: (nD stack of) Vector(s) for first dot product operand
    :param b: (nD stack of) Vector(s) for second dot product operand
    :param array: If true and passed a single vector for each operand, return a
        numpy 1D array result. Otherwise you will get a scalar result if you pass
        in single vector operands. No effect if passed stacks as either operand
    :return: Dot product(s) of inputs

    This uses numpy broadcasting to calculate the result, so the operands do not
    have to be the same size, just broadcast-compatible. In this case, the result
    may be larger than either input.

    If one input has more components than the other, the result will be equivalent
    to the result of the shorter input having the same number of components, all
    of which are zero. Equivalently, the result is equivalent to the longer input
    being truncated to match the length of the shorter input. Note that this only
    applies to the vector component -- all other axes of a stack are subject to
    numpy broadcast rules.
    """
    n = np.min((vncomp(a), vncomp(b)))
    c = vcomplimit(a, n) * vcomplimit(b, n)
    result = np.sum(c, axis=vindex(c))
    if result.size == 1 and not array:
        result = result.ravel()[0]
    if np.isscalar(result) and array:
        result = np.array([result])
    return result

def vlength(v, array=False):
    """
    Compute the length of a vector as the square root of the vector's dot product with itself

    :param v: (Stack of) Vector(s) to compute the length of
    :param array: Passed to vdot

        If true and passed a single vector, return a numpy 1D array result. Otherwise you will get a scalar result if you pass a single vector.
        If true and passed a stack of vectors, result will have the shape of a stack of 1D vectors.
        IE if you pass a vector of shape (3072,3,4096), the answer will have shape (3072,1,4096)

    """
    return np.sqrt(vdot(v, v, array))


def vnormalize(v):
    """
    Compute the normalized (unit-length) vector(s) parallel to the given vector(s)
    :param v: Vector or stack of vectors
    :return: Vector or stack of vectors with same direction but unit length
    """
    return v/vlength(v)


def vangle(a, b, array=False, deg:bool=False):
    """
    Compute the angle between two vectors

    :param a: (stack of) first  vector operand(s)
    :param b: (stack of) second vector operand(s)
    :param array: Passed to vdot and vlength
    :param deg: If true, return angle in degrees. Otherwise radians by default.
    :return: Angle(s) between two (stacks of) vectors in radians

    Note - using true real numbers, it is impossible for the dot product to be
           greater than the product of the input vector lengths, and therefore
           impossible for the argument to arccos to be outside of [-1,1].
           However, it can and does happen with limited-precision floating
           point numbers. This has been observed operationally, and must be
           accounted for. It is assumed but not checked that if the argument
           is out of range, it is only out by a small amount.
    """
    arg=vdot(a, b, array) / vlength(a, array) / vlength(b, array)
    if np.ndim(arg)>0:
        arg[np.where(arg<-1)]=-1
        arg[np.where(arg> 1)]= 1
    else:
        if arg<-1:
            arg=-1
        if arg> 1:
            arg= 1
    result=np.arccos(arg)
    if deg:
        result=np.rad2deg(result)
    return result


def vcomp(comps):
    """
    Compose stacks of vector components into a single stack of vectors (inverse of vdecomp())

    :param comps: Iterable of components. All components must be the same size. m will be length of comps
                  unless adjusted by l, minlen, or maxlen below.
    :return: nD stack of m-element vectors

    """
    try:
        if len(comps[0].shape) >= 2:
            axis = -2
        elif len(comps[0].shape)==0:
            return np.stack([np.array([x]) for x in comps],axis=0)
        else:
            axis = 0
    except AttributeError:
        #This case handles things that aren't already numpy arrays, and is triggered by
        #any of the input comps not having a .shape attribute.
        comps=[np.array([x]) for x in comps]
        axis=0
    return np.stack(comps, axis=axis)


def vdecomp(v, m=None, minlen=None, maxlen=None, array=False):
    """
    Decompose a vector into components. an nD stack of m-element vectors will return a tuple with up to m elements,
    each of which will be an nD stack of scalars

    :param v: nD stack of m-element vectors, a numpy (n+1)D array with shape
        (n_stack0,n_stack1,...,n_stackn-2,m,n_stackn-1)
    :param minlen: If passed, this will pad out the returned vector components with zero scalars
        such that the returned tuple has minlen components. We do zero scalars rather than zero arrays
        of the same size as the other components to save memory, since a scalar is compatible by
        broadcasting with an array of any size.
    :param maxlen: If passed, this will restrict the returned vector components to the given
        size, even if the input vector has more components.
    :param m: If passed, treat the input as if it were an nD stack of m-element vectors. If the actual
              stack has more components, don't return them. If it has less, return scalar zeros for the
              missing components
    :param array: If true and passed a single vector, return a
        numpy 1D array result. Otherwise you will get a scalar result if you pass
        in single vector operands. No effect if passed stacks as either operand
    :return: A tuple. Each element is a vector component. Vector components pulled from the vector will be
        an nD stack of scalars, a numpy nD array with shape (n_stack0,n_stack1,...,n_stackn-2,n_stackn-1).
        Vector components which are made up will be scalar zeros.

    Note: If you pass maxlen<minlen, the result is still well-defined, since the maxlen is used first,
          then the minlen. If you pass a vector with m=4, a minlen of 7, and a maxlen of 2, you will get
          a result with the first two components of the vector, followed by 5 zeros. I'm not sure if this
          is useful, but there it is.
    Example:
        v=np.zeros((24,3,50)) #Suitable for holding multiple trajectories

        #OR

        v0=np.zeros((3,50)) #Initial conditions for 50 trajectories
        t=np.arange(24)     #Time steps
        v=rk4(x0=v0,t=t)    #Numerically integrate multiple trajectories. Result shape will be (t.size,)+v0.shape,
                            #IE (24,3,50)
        x,y,z=vdecomp(v) #after this, x, y, and z are each numpy arrays of shape (24,50)
    """
    if maxlen is None and m is not None:
        maxlen = m
    if minlen is None and m is not None:
        minlen = m
    ndStack = len(v.shape) > 2
    efflen = v.shape[-2 if ndStack else 0]
    if maxlen is not None and maxlen < efflen:
        efflen = maxlen
    result = tuple([v[..., i, :] if ndStack else v[i, ...] for i in range(efflen)])
    if minlen is not None and minlen > efflen:
        result = result + tuple([np.zeros(1) for i in range(minlen - efflen)])
    if result[0].size == 1 and not array:
        result = tuple([x.ravel()[0] for x in result])
    if np.isscalar(result[0]) and array:
        result = np.array([result])
    return result


def vcross(a, b):
    """
    Compute the three-dimensional cross-product of two vectors or stack of vectors
    :param a: (nD stack of) Vector(s) for first cross product operand
    :param b: (nD stack of) Vector(s) for second cross product operand
    :return: Cross product(s) of inputs
    This uses numpy broadcasting to calculate the result, so the operands do not
    have to be the same size, just broadcast-compatible. In this case, the result
    may be larger than either input.
    If either of the input vectors have fewer than three components, the extra components
    are made up and assumed to be zero. If either input has more than three components,
    the extra components are ignored. The result will always have three components.
    Inputs a and b may be row vectors, but in this case, the result is a column vector.
    """
    (ax, ay, az) = vdecomp(a, m=3)
    (bx, by, bz) = vdecomp(b, m=3)
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return vcomp((cx,cy,cz))


def vncross(a, b):
    """
    Normalized cross product
    :param a: First cross product factor
    :param b: Second cross product factor
    :return: Unit vector in same direction as vcross(a,b)
    """
    return vnormalize(vcross(a, b))


def vforce_proj(a,b,d):
    """
    Force a vector to have a given length projection with another vector.
    Calculate a scalar constant k such that dot(k*a,b)=d and return k*a

    :param a: Vector to scale
    :param b: Vector to use as reference
    :param d: Required dot product
    :return: A vector k*a with the same direction as a which has the
             given dot product with b

    """
    # Since the dot product is linear:
   #vdot(k*a,b)=d
   #k*vdot(a,b)=d
    k=d/vdot(a,b)
    #Stretch k to be broadcastable with a
    try:
        return k[...,np.newaxis,:]*a
    except:
        return k*a

def rv(sv):
    """
    Position part of state vector
    :param sv: Stack of state vectors, can be one in stack IE column vector
    :return: Position part, will be stack matching sv
    """
    return sv[:3,:]

def vv(sv):
    """
    Velocity part of state vector
    :param sv: Stack of state vectors, can be one in stack IE column vector
    :return: Position part, will be stack matching sv
    """
    return sv[3:,:]


def where(condition:bool|np.ndarray, x:Number|np.ndarray, y:Number|np.ndarray)->Number|np.ndarray:
    """
    A wrapper around np.where to handle scalar inputs and outputs.

    This function returns a scalar where that makes sense,
    unlike np.where() which always returns an ndarray, even
    a single-element one that it considers a scalar.

    :param condition: The condition to evaluate.
    :param x: Value to return if condition is True.
    :param y: Value to return if condition is False.
    :return: Scalar or array based on input types.
    """
    is_scalar = not isinstance(condition, np.ndarray) and not isinstance(x,np.ndarray) and not isinstance(y,np.ndarray)
    result = np.where(condition, x, y)
    if is_scalar and isinstance(result,np.ndarray):
        return result.item()
    return result


def select(condlist:list[bool|np.ndarray], choicelist:list[Number|np.ndarray], default:Number|np.ndarray=0)->Number|np.ndarray:
    """
    A wrapper around np.select to handle scalar inputs and outputs.

    This function returns a scalar where that makes sense,
    unlike np.select() which always returns an ndarray, even
    a single-element one that it considers a scalar.

    :param condlist: The condition to evaluate.
    :param x: Value to return if condition is True.
    :param y: Value to return if condition is False.
    :return: Scalar or array based on input types.
    """
    is_scalar = np.all([not isinstance(cond, np.ndarray) for cond in condlist]+
                       [not isinstance(choice, np.ndarray) for choice in choicelist]+
                       [not isinstance(default, np.ndarray)])
    result = np.select(condlist, choicelist, default=default)
    if is_scalar and isinstance(result,np.ndarray):
        return result.item()
    return result

