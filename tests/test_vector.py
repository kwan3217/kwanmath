"""
Describe purpose of this script here

Created: 1/23/25
"""
import numpy as np
import pytest

from kwanmath.vector import where, select, vcomp, vangle


@pytest.mark.parametrize(
    "cond,a,b,out,t",
    [(True,0,1,0,int),
     (False,0,1,1,int),
     (np.array((True,False)),0,1,np.array((0,1)),np.ndarray)]
)
def test_where(cond,a,b,out,t):
    w=where(cond,a,b)
    print(type(w),w)
    assert np.all(w==out)
    assert type(w)==t

@pytest.mark.parametrize(
    "condlist, choicelist, default, out, t",
    [
        # Scalar test
        ([True], [1], 0, 1, int),
        ([False], [1], 0, 0, int),
        # Array test
        ([np.array([True, False])], [np.array([1, 2])], 0, np.array([1, 0]), np.ndarray),
        # Mixed types with scalar output
        ([1 < 2, 2 < 3], [1, 2], 3, 1, int),
        # Mixed types with array output
        ([np.array([1 < 2, 2 < 3]), np.array([2 < 3, 3 < 4])],
         [np.array([1, 2]), np.array([3, 4])],
         5, np.array([1, 2]), np.ndarray)
    ]
)
def test_select(condlist, choicelist, default, out, t):
    s = select(condlist, choicelist, default)
    print(type(s), s)
    assert np.all(s == out)
    assert type(s) == t


@pytest.mark.parametrize(
    "a,b,deg,result",
    [(vcomp((1,0,0)),vcomp((0,1,0)),True,90.0),
     (vcomp((1,0,0)),vcomp((1,1,0)),True,45.0),
     (vcomp((1,0,0)),vcomp((1,1,0)),False,np.pi/4.0),
     ]
)
def test_vangle(a:np.ndarray,b:np.ndarray,deg:bool,result:np.ndarray):
    assert np.allclose(vangle(a,b,deg=deg),result)