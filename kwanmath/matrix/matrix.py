"""
Matrix routines that are not supplied by numpy,
or are not using the exact convention that EXI
uses in other places

* rot_x, rot_y, rot_z: Return a rotation matrix
  about one of the primary axes.
"""

import numpy as np

def rot_axis(axis,theta):
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
    c=np.cos(theta)
    s=np.sin(theta)
    result=np.identity(3)
    result[(axis+1)%3,(axis+1)%3]=c; result[(axis+1)%3,(axis+2)%3]=-s
    result[(axis+2)%3,(axis+1)%3]=s; result[(axis+2)%3,(axis+2)%3]= c
    return result


def rot_x(theta):
    """
    Rotation matrix around X axis
    :param theta: Rotation angle in radians, right-handed
    :return: Rotation matrix in form of (3,3) 2D numpy array
    """
    return rot_axis(0,theta)

def rot_y(theta):
    """
    Rotation matrix around Y axis
    :param theta: Rotation angle in radians, right-handed
    :return: Rotation matrix in form of (3,3) 2D numpy array
    """
    return rot_axis(1,theta)

def rot_z(theta):
    """
    Rotation matrix around Z axis
    :param theta: Rotation angle in radians, right-handed
    :return: Rotation matrix in form of (3,3) 2D numpy array
    """
    return rot_axis(2,theta)

