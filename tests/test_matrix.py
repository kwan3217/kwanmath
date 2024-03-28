"""
Tests for emmexipy.python_lib.matrix
"""
import numpy as np
import pytest

from kwanmath.geodesy import llr2xyz
from kwanmath.matrix import rot_axis, euler_matrix, point_toward
from kwanmath.vector import vcomp


@pytest.mark.parametrize(
    "axis,angle,xb_r,yb_r,zb_r",[
    (0,np.pi/2,np.array([[1],[0],[0]]),np.array([[0],[0],[1]]),np.array([[0],[0],[-1]]))
        ]
)
def test_rot_axis(axis,angle,xb_r,yb_r,zb_r):
    """
    Test that the rotation function returns a matrix which does what we intend
    :param axis: Axis to rotate around, 0=x, 1=y, 2=z
    :param angle: Angle in radians to rotate
    :param xb_r: transformed body x axis in reference frame
    :param yb_r: transformed body y axis in reference frame
    :param zb_r: transformed body z axis in reference frame
    :return: None, but raises an exception if the test fails
    """
    M=rot_axis(axis,angle)
    assert np.allclose(M@np.array([[1],[0],[0]]),xb_r)


def test_euler_matrix():
    yaw=-0.103420
    pitch=-89.981545
    roll=-111.916408
    #yaw=0
    #pitch=-90
    #roll=-112.5
    # The following data is copied from the kernel emm_fk_rev015 from TKFRAME_-62102_MATRIX
    # (the correct one for vis). This data is NOT transposed from how it is in Spice. Spice
    # effectively does a transpose, and wants the matrix that transforms from the given frame
    # to the relative frame. Therefore we compare the transpose of the matrix as printed
    # in the kernel
    M_ref=np.array([[-3.73253567e-01,  3.74901147e-04,  9.27729289e-01],
                    [ 9.27727609e-01, -1.79479360e-03,  3.73253617e-01],
                    [ 1.80501580e-03,  9.99998319e-01,  3.22106969e-04]]).T
    #print(M_ref)
    M_test=euler_matrix((roll,pitch,yaw),deg=True)
    #print(M_test)
    #print(M_test-M_ref)
    assert np.all(np.isclose(M_test,M_ref))


@pytest.mark.parametrize(
    "p_b,p_r,t_b,t_r,ref_Mrb",[
        (llr2xyz(lat=-13.0,lon=0.0,r=1.0,deg=True), #p_b, 13deg below nose
         llr2xyz(lat=30.0,heading=80.0,r=1.0,deg=True), #p_r, 30deg above horizon at azimuth 80deg east of north
         vcomp((0.0,0.0,1.0)), #t_b, tail
         vcomp((0.0,0.0,-1.0)),#t_r, heads down
         np.array((( 0.941776, 0.173648, 0.287930), # ref_Mrb, matrix which transforms to reference from body
                   ( 0.166061,-0.984808, 0.050770),
                   ( 0.292372, 0.000000,-0.956305)))
        )
    ]
)
def test_point_toward(p_b:np.ndarray,p_r:np.ndarray,t_b:np.ndarray,t_r:np.ndarray,ref_Mrb:np.ndarray):
    """
    Test the point_toward() function

    :return: None, but raises an exception if the test fails

    \image html Space_Shuttle_Coordinate_System.jpg
    \f$
        \def\M#1{{[\mathbf{#1}]}}
        \def\MM#1#2{{[\mathbf{#1}{#2}]}}
        \def\T{^\mathsf{T}}
        \def\operatorname#1{{\mbox{#1}}}
    \f$

    The space shuttle has a thrust axis 13&deg; below the X axis, so:
    \f$\hat{p}_b=\begin{bmatrix}\cos 13^\circ \\ 0 \\ -\sin 13^\circ \end{bmatrix}
       =\begin{bmatrix}0.974370 \\ 0.000000 \\ -0.224951 \end{bmatrix}\f$

    The heads-up vector is \f$\hat{t}_b=\hat{z}_b\f$. At a particular instant,
    the guidance command says to point the thrust vector 30&deg; above the horizon
    at an azimuth of 80&deg; east of North. We'll take the local topocentric horizon
    frame as the reference frame, with \f$\hat{x}_r\f$ in the horizon plane pointing
    east, \f$\hat{y}_r\f$ pointing north, and \f$\hat{z}_r\f$ pointing up. In this
    frame, the guidance command is:

    \f$\hat{p}_r=\begin{bmatrix}\cos 30^\circ \sin 80^\circ \\
                                \cos 30^\circ \cos 80^\circ \\
                                \sin 30^\circ\end{bmatrix}=\begin{bmatrix}0.852869 \\
                                                                          0.150384 \\
                                                                          0.500000\end{bmatrix}\f$

    The vehicle is also commanded to the heads-down attitude, which means that
    \f$\hat{t}_r=-\hat{z}_r\f$. These are all the inputs we need.

    \f$\hat{s}_b=\operatorname{normalize}(\hat{p}_b \times \hat{t}_b)=\begin{bmatrix} 0 \\
                                                                                     -1 \\
                                                                                      0 \end{bmatrix}\f$

    \f$\hat{u}_b=\operatorname{normalize}(\hat{p}_b \times \hat{s}_b)=\begin{bmatrix} -0.224951 \\
                                                                                       0.000000 \\
                                                                                      -0.974370 \end{bmatrix}\f$
    \f$\hat{s}_r=\operatorname{normalize}(\hat{p}_r \times \hat{t}_r)=\begin{bmatrix} -0.173648 \\
                                                                                       0.984808 \\
                                                                                       0.000000 \end{bmatrix}\f$

    \f$\hat{u}_r=\operatorname{normalize}(\hat{p}_r \times \hat{s}_r)=\begin{bmatrix} -0.492404 \\
                                                                                      -0.086824 \\
                                                                                      -0.866025 \end{bmatrix}\f$

    \f$\M{R}=\begin{bmatrix}\hat{p}_r && \hat{s}_r\ && \hat{u}_r \end{bmatrix}=\begin{bmatrix}0.852869&&-0.173648&&-0.492404\\
                                                                                              0.150384&& 0.984808&&-0.086824\\
                                                                                              0.500000&& 0.000000&& 0.866025\end{bmatrix}\f$

    \f$\M{B}=\begin{bmatrix}\hat{p}_b && \hat{s}_b\ && \hat{u}_b \end{bmatrix}=\begin{bmatrix}0.974370&& 0.000000&&-0.224951\\
                                                                                              0.000000&&-1.000000&&-0.000000\\
                                                                                             -0.224951&& 0.000000&&-0.974370\end{bmatrix}\f$
    \f$\M{M_{br}}=\M{R}\M{B}^{-1}=\begin{bmatrix}0.941776&& 0.173648&& 0.287930\\
                                                 0.166061&&-0.984808&& 0.050770\\
                                                 0.292372&& 0.000000&&-0.956305\end{bmatrix}\f$

    There is the solution, but does it work?

    \f$\begin{eqnarray*}\M{M_{br}}\hat{p}_b&=&\begin{bmatrix} 0.852869\\ 0.150384\\ 0.500000\end{bmatrix}&=&\hat{p}_r \\
                        \M{M_{br}}\hat{s}_b&=&\begin{bmatrix}-0.173648\\ 0.984808\\ 0.000000\end{bmatrix}&=&\hat{s}_r \\
                        \M{M_{br}}\hat{u}_b&=&\begin{bmatrix}-0.492404\\-0.086824\\ 0.866025\end{bmatrix}&=&\hat{u}_r \\
                        \M{M_{br}}\hat{t}_b&=&\begin{bmatrix} 0.287930\\ 0.050770\\-0.956305\end{bmatrix}, \operatorname{vangle}(\M{M_{br}}\hat{t}_b,\hat{t}_r)=17^\circ\end{eqnarray*}\f$

    That's a decisive yes.
    """
    test_Mrb=point_toward(p_b=p_b,p_r=p_r,t_b=t_b,t_r=t_r)
    assert np.allclose(test_Mrb,ref_Mrb)
    assert np.allclose(p_r,test_Mrb @ p_b)
    assert np.allclose(p_r,ref_Mrb @ p_b)

