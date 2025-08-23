"""
Tests for emmexipy.python_lib.matrix
"""
import numpy as np
import pytest

from kwanmath.geodesy import llr2xyz
from kwanmath.matrix import rot_axis, euler_matrix, point_toward, aa_to_m, m_to_aa, slerp
from kwanmath.vector import vcomp, vnormalize, vlength


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


def test_axis_angle():
    ref_aa=45.0*vnormalize(vcomp((1.0,1.0,1.0)))
    M_rb=aa_to_m(ref_aa,deg=True)
    test_aa=m_to_aa(M_rb,deg=True)
    assert np.allclose(test_aa,ref_aa)


def test_small_aa():
    ref_aa=1e-99*vnormalize(vcomp((1.0,1.0,1.0)))
    assert vlength(ref_aa)!=0.0, "ref_aa too small, indistinguishable from 0"
    M_rb=aa_to_m(ref_aa,deg=True)
    trace=np.trace(M_rb)
    assert trace==3.0, f"ref_aa not small enough to test this case, {trace.hex()}!={(3.0).hex()}"
    test_aa=m_to_aa(M_rb,deg=True)
    print(ref_aa,test_aa)
    assert np.isclose(vlength(ref_aa),vlength(test_aa),atol=0)


def test_slerp_trivial():
    M0=np.eye(3)
    M1=np.array([[ 0.0,-1.0, 0.0],
                 [ 1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 1.0]])
    #Trivial cases: for t=0, we should get M0, and for t=1, we should get M1
    s=slerp(M0,M1)
    assert np.allclose(s(0.0),M0)
    assert np.allclose(s(1.0),M1)
    # Interesting case: for t=0.5, should have 0.707 in the upper left 2x2. Specifically something like:
    # x_ct is between +x_u and +y_u, so [.707,.707,0]^T
    # y_ct is between +y_u and -x_u, so [-.707,.707,0]^T
    # z_ct is still z_u
    s2o2=np.sqrt(2.0)/2.0
    M05_ref=np.array([[s2o2,-s2o2,0.0],
                      [s2o2, s2o2,0.0],
                      [   0,    0,1.0]])
    assert np.allclose(s(0.5),M05_ref)


def test_slerp_broadcast():
    M0=np.eye(3)
    M1=np.array([[ 0.0,-1.0, 0.0],
                 [ 1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 1.0]])
    t=np.arange(0,1,0.01)
    Mt=slerp(M0,M1,t)
    assert Mt.shape==(100,3,3)


def test_random_slerp():
    M0=np.array([[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]])
    M1=np.array([[-0.23954104, -0.96908537,  0.05910691],
                 [ 0.94077109, -0.21663513,  0.2608045 ],
                 [-0.23993719,  0.11807946,  0.9635805 ]])
    M05_ref = np.array([[ 6.15975315e-01,-7.82164407e-01, 9.37723430e-02],
                        [ 7.73392300e-01, 6.23071841e-01, 1.16815373e-01],
                        [-1.49795733e-01, 5.67422173e-04, 9.88716803e-01]])
    M05_test=slerp(M0,M1,0.5)
    assert np.allclose(M05_ref,M05_test)