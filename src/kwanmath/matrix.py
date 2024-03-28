"""
Matrix routines that are not supplied by numpy,
or are not using the exact convention that EXI
uses in other places

* rot_x, rot_y, rot_z: Return a rotation matrix
  about one of the primary axes.
* point_toward: Compute the point-toward transformation
"""

import numpy as np
from .vector import vnormalize, vcross


def rot_axis(axis,theta=None,*,c=None,s=None):
    """
    Rotate an object around the given reference axis
    by the given amount
    :param axis: 0 for x axis, 1 for y, 2 for z
    :param theta: angle to rotate in a right-handed sense in radians
    :return: Mrb, a 3x3 matrix that transforms to reference frame
             from rotated body frame

    This returns a rotation matrix which transforms to
    the reference frame from the body frame of a body
    which has been rotated in a right-handed sense
    around the given axis by the given amount. We will
    describe this in excruciating detail so that this
    convention can be properly compared to any other
    convention used in any other source.

    Body frame -- Imagine holding something
    with an obvious orientation, like a mannequin. 
    It has a body frame such that it is
    standing with its face pointing towards +X, left
    side towards +Y, head towards +Z and feet towards
    -Z. No matter how you turn the figure, the body
    frame stays attached to the figure and all of the
    above relations stay true.

    Right-handed rotation -- Hold your right hand
    with the thumb sticking straight out, and the fingers
    curled, as if you are holding a bar. If the thumb points
    in the direction of the axis, then the fingers indicate
    the positive right-handed rotation direction.

    Rotation example -- We are going to rotate the
    figure by 90deg around the Z axis. Point your
    right thumb towards +Z (along the head-feet axis)
    and curl your fingers around and rotate the figure
    that way. This will bring the face to point at +Y
    in the external reference frame (but still at +X in
    the body frame). The head will still point towards
    +Z and the feet towards -Z, but now the left side
    will be towards -X and the right side towards +X.

    Body-to-reference -- This rotation is now described
    with a matrix. When any vector in the body frame
    is multiplied by this matrix, you get the corresponding
    vector in the external reference frame. Since the face
    now points to reference +Y, we want a matrix which
    transforms +X_body to +Y_reference. Similarly
    +Y_body (left side) will transform to -X_reference,
    and +Z_body won't change and will transform to
    +Z_reference.

    Note:
    * Rotations combine in complicated ways, such
    that order matters. Rotating around X then Z will
    result in a different final orientation than rotating
    around Z then X.
    * This function can only be used to rotate around
    one and only one of the principal body axes (X,Y,Z),
    not around some arbitrary axis at some angle to these.
    * Rotations can be combined with matrix multiplication.
    Each rotation creates an intermediate frame. For
    instance if you rotate 60deg around the Z axis, this
    creates a new intermediate frame which will itself be
    rotated by subsequent rotations. If you do that
    60degZ *then* 90degY, you will be rotating around the
    Y axis of the intermediate frame, which is 60deg away
    from the original Y body frame.
    * Three axis rotations that aren't all about the same
    axis are sufficient to rotate a body to any arbitrary
    orientation. Look up "Euler Angles" for more
    information. These functions are ideal for
    implementing Euler angle rotation. They are not ideal
    for other orientation descriptions such as
    "quaternion" or "axis and angle".
    * These have been carefully tested such that the
    actual result of the functions match the English
    description above, but there are many conventions out
    in the world. These may not actually match the
    rot_x,y,z used in other references. Please check
    carefully if the rotation sense really is the same
    before using these functions.
    """
    if c is None:
        c=np.cos(theta)
    if s is None:
        s=np.sin(theta)
    result=np.identity(3)
    result[(axis+1)%3,(axis+1)%3]=c; result[(axis+1)%3,(axis+2)%3]=-s
    result[(axis+2)%3,(axis+1)%3]=s; result[(axis+2)%3,(axis+2)%3]= c
    return result


def rot_x(theta=None,*,c=None,s=None):
    """
    Rotation matrix around X axis
    :param theta: Rotation angle in radians, right-handed
    :return: Rotation matrix in form of (3,3) 2D numpy array
    """
    return rot_axis(0,theta,c=c,s=s)

def rot_y(theta=None,*,c=None,s=None):
    """
    Rotation matrix around Y axis
    :param theta: Rotation angle in radians, right-handed
    :return: Rotation matrix in form of (3,3) 2D numpy array
    """
    return rot_axis(1,theta,c=c,s=s)

def rot_z(theta=None,*,c=None,s=None):
    """
    Rotation matrix around Z axis
    :param theta: Rotation angle in radians, right-handed
    :return: Rotation matrix in form of (3,3) 2D numpy array
    """
    return rot_axis(2,theta,c=c,s=s)


def euler_matrix(thetas:tuple,axes:tuple=(2,0,2),deg:bool=False)->np.array:
    """
    Return a rotation matrix which performs a rotation indicated by three Euler angles

    :param thetas: Angles, given in order that the rotation will be performed
    :param axes: Axis identifier for each theta, same order as thetas, x=0,y=1,z=2
                 Default is useful for aircraft -- roll first, then pitch, then yaw. If done in this order,
                 nose will be pointing above or below horizon by "pitch" and with azimuth "yaw", no matter
                 the roll.
    :param deg: True if the angles are in degrees, default is radians
    :return: Rotation matrix representing the given Euler angles and axes

    Example: Perform a 20deg roll (rotation around z), followed by a 90deg pitch (rotation
             around x), followed by a 10deg yaw (rotation around z)
    M=euler_matrix(thetas=(20,90,10),deg=True)

    Note that frequently you will see rotations specified like this in various papers:

    M=rx(thetax) @ ry(thetay) @ rz(thetaz)

    Note that this rotates first around Z, then y, then x. To do this, we would use:

    M=euler_matrix(thetas=(thetaz,thetay,thetax),axes=(2,1,0))

    where the thetas and axes are specified in the order they are applied, or right-to-left
    if reading a matrix multiplication.

    """
    M=np.identity(3)
    for theta,axis in zip(thetas,axes):
        M=rot_axis(axis=axis,theta=np.deg2rad(theta) if deg else theta) @ M
    return M


def point_toward(*,p_b:np.array,
                   p_r:np.array,
                   t_b:np.array,
                   t_r:np.array)->np.array:
    """
    Generate the point-toward matrix. This is a matrix M_rb that transforms a
    vector in a body frame to a vector in a reference frame. The point vector
    p_b is transformed to be parallel to p_r, and the toward vector t_b is
    transformed to be as close as possible to t_r.
    :param p_b: Point body vector
    :param p_r: Point reference vector
    :param t_b: Toward body vector
    :param t_r: Toward reference vector
    :return: Matrix M_rb that transforms to reference frame from body.
    If vlength(p_r)==vlength(p_b), then to within floating-point precision,
    p_r=M_rb @ p_b

    """
    #Force all inputs to unit-length column vectors
    ph_r=vnormalize(p_r.reshape(3,1))
    ph_b=vnormalize(p_b.reshape(3,1))
    th_r=vnormalize(t_r.reshape(3,1))
    th_b=vnormalize(t_b.reshape(3,1))
    sh_r=vnormalize(vcross(ph_r,th_r))
    sh_b=vnormalize(vcross(ph_b,th_b))
    uh_r=vnormalize(vcross(ph_r,sh_r))
    uh_b=vnormalize(vcross(ph_b,sh_b))
    R=np.hstack((ph_r,sh_r,uh_r))
    B=np.hstack((ph_b,sh_b,uh_b))
    M_rb=R @ B.T
    return M_rb


def Mtrans(M,*vs):
    """
    Transform a matrix or stack of matrices against a vector or stack of vectors
    :param M: Matrix [rows,columns] or stack of matrices [stack,rows,columns]
    :param v: Column vector [rows,1] or stack of vectors [rows,stack]
    :return: Transformed vector(s)
    If M and v are both singles, return a single column vector
    If M is single and v is stack, return a stack of vectors, each one transformed against the (one and only) matrix
    If M is stack, v must be stack, return a stack of vectors, each one transformed against the corresponding matrix
    """
    if len(M.shape)>2:
        result=tuple([ (M @ (v.transpose().reshape(v.shape[1], v.shape[0], 1)))[:, :, 0].transpose() for v in vs])
    else:
        result=tuple([M @ v for v in vs])
    if len(vs)==1:
        result=result[0]
    return result

def Mr(Ms):
    """
    Get position part of state transformation matrix
    :param Ms: Stack of state transformation matrices
    :return: Position part, IE upper 3x3 of each matrix
    """
    return Ms[...,:3,:3]

def Mv(Ms):
    """
    Get position part of state transformation matrix
    :param Ms: Stack of state transformation matrices
    :return: Position part, IE upper 3x3 of each matrix
    """
    return Ms[...,3:,3:]


def isrot(R):
    """
    Checks if a matrix is a valid rotation matrix.

    :param R:
    :return:
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
