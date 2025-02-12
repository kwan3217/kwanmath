"""
Matrix routines that are not supplied by numpy,
or are not using the exact convention that EXI
uses in other places

* rot_x, rot_y, rot_z: Return a rotation matrix
  about one of the primary axes.
* point_toward: Compute the point-toward transformation
"""

import numpy as np
from kwanmath.vector import vnormalize, vcross, vcomp, vlength, vdecomp


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


def m_to_aa(M_rb:np.ndarray,deg:bool=False)->np.ndarray:
    """
    Given a matrix which transforms a vector from body coordinates to
    reference coordinates, calculate the axis-and-angle with vector
    magnitude encoding of the angle.
    :param M_rb: Matrix which transforms a vector v_b from body
                 to reference. Must be a valid rotation matrix
    :param deg:  If True, the returned vector length represents an angle
                 in degrees; otherwise, in radians. Default is False (radians).
    :return: Vector with direction as axis of rotation and length
             as angle
    """
    # Check that M_rb is in fact in SO(3)
    if not np.allclose(M_rb @ M_rb.T, np.eye(3)) or not np.isclose(np.linalg.det(M_rb), 1):
        raise ValueError("Input matrix is not a valid rotation matrix")

    # Compute the trace of the matrix
    trace = np.trace(M_rb)

    # Case for theta = 0 (no rotation)
    if trace == 3:
        # In this case, the matrix is:
        #       [ 1 -z  y]
        # M_rb= [ z  1 -x]
        #       [-y  x  1]
        # where [[x],[y],[z]] has magnitude ~sin(theta) and correct direction
        # but the angle is so small that sin(theta)~=theta in radians. with error O(epsilon**3)
        result=vcomp((M_rb[2, 1],
                      M_rb[0, 2],
                      M_rb[1, 0]))  # Use the (noninverted) off-diagonal values
        if deg:
            result=np.rad2deg(result)
        return result

    # Angle of rotation
    theta = np.arccos((trace - 1) / 2.0)

    # Axis of rotation
    if theta != 0:
        sin_theta = np.sin(theta)
        axis = vcomp((
            (M_rb[2, 1] - M_rb[1, 2]) / (2 * sin_theta),
            (M_rb[0, 2] - M_rb[2, 0]) / (2 * sin_theta),
            (M_rb[1, 0] - M_rb[0, 1]) / (2 * sin_theta)
        ))
    else:
        # If theta is zero, any axis is valid, here we just use [1, 0, 0] for simplicity
        axis = vcomp((1., 0., 0.))

    # Normalize axis to unit length
    axis = axis / vlength(axis)

    # Combine axis and angle
    if deg:
        theta=np.rad2deg(theta)
    return theta * axis


def aa_to_m(aa_vector,deg:bool=False):
    """
    Convert an axis-angle representation, where the angle is encoded in the vector's magnitude,
    back to a rotation matrix. This is intended to be the inverse of m_to_aa, such that
    m_to_aa(aa_to_m(aa))==aa.

    :param aa_vector:  Vector where the direction indicates the axis of rotation and the magnitude
                       represents the angle of rotation.
    :param deg:        If True, the vector length represents an input angle in degrees; otherwise,
                       in radians. Default is False (radians).
    :return:
    """
    # Extract the angle from the length of the vector
    theta = vlength(aa_vector)

    # If theta is zero, return identity matrix (no rotation)
    if theta == 0:
        return np.eye(3)

    # Normalize the vector to get the axis of rotation. We do this
    # before converting theta to radians, because that's what is
    # needed to get a unit vector.
    axis = aa_vector / theta
    if deg:
        theta=np.deg2rad(theta)

    # Components of the axis
    x, y, z = vdecomp(axis)

    # Compute the cosine and sine of theta
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c

    # Construct the rotation matrix
    M_rb = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
    ])

    return M_rb