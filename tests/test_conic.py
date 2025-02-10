"""
Describe purpose of this script here

Created: 2/5/25
"""
import numpy as np
import pytest

from kwanmath.conic import fit_conic, eval1_conic, eval2_conic, vector_to_conic

test_x=np.array((603, 549, 531, 555, 601))
test_y=np.array((230, 257, 306, 361, 380))

@pytest.mark.parametrize(
    "x,y,scale",
    [(test_x,test_y,1000.0)]
)
def test_fit_conic(x,y,scale):
    Ap,Bp,Cp,Dp,Ep=fit_conic(np.row_stack((x,y)),scale=scale)
    xp=np.array(x)/scale
    yp=np.array(y)/scale
    assert np.allclose(Ap*xp**2+Bp*xp*yp+Cp*yp**2+Dp*xp+Ep*yp,1)


@pytest.mark.parametrize(
    "x,y,scale",
    [(test_x,test_y,1000.0)]
)
def test_eval1_conic(x,y,scale):
    Ap,Bp,Cp,Dp,Ep=fit_conic(np.row_stack((x,y)),scale=scale)
    A=Ap/(scale**2)
    B=Bp/(scale**2)
    C=Cp/(scale**2)
    D=Dp/scale
    E=Ep/scale
    x,y0,y1=eval1_conic(Ap,Bp,Cp,Dp,Ep,scale=scale)
    assert np.allclose(A*x**2+B*x*y0+C*y0**2+D*x+E*y0,1)
    assert np.allclose(A*x**2+B*x*y1+C*y1**2+D*x+E*y1,1)


@pytest.mark.parametrize(
    "x,y,scale",
    [(np.array((-1.0,0.0,1.0,0.0,2.0)),np.array((0.0,1.0,0.0,-1.0,2.0)),1.0),
     (test_x,test_y,1000.0)]
)
def test_eval2_conic(x,y,scale):
    import matplotlib.pyplot as plt
    plt.plot(x,y,'+')
    plt.axis('scaled')
    plt.pause(1)
    Ap,Bp,Cp,Dp,Ep=fit_conic(np.row_stack((x,y)),scale=scale)
    A=Ap/(scale**2)
    B=Bp/(scale**2)
    C=Cp/(scale**2)
    D=Dp/scale
    E=Ep/scale
    r=eval2_conic(Ap,Bp,Cp,Dp,Ep,scale=scale)
    x=r[0,:]
    y=r[1,:]
    plt.plot(x,y,'-')
    plt.axis('scaled')
    plt.pause(1)
    assert np.allclose(A*x**2+B*x*y+C*y**2+D*x+E*y,1)


def test_vector_to_conic():
    """
    Test vector_to_conic() by round trip. Make sure that cv+-av and cv+-bv satisfy conic equation
    :return: None, but raises an exception if the test fails
    """
    cv=np.array([[3.0],[4.0]])
    av=np.array([[2.0],[1.0]])
    bv=np.array([[-1.0],[2.0]])
    A,B,C,D,E,rr=vector_to_conic(cv=cv,av=av,bv=bv,scale=1,get_points=True)
    r=np.hstack((cv+av,cv-av,cv+bv,cv-bv,rr))
    x=r[0,:]
    y=r[1,:]
    assert np.allclose(A*x**2+B*x*y+C*y**2+D*x+E*y,1)


