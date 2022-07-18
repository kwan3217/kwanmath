"""
Foo all the Bars
"""

import pytest
from kwanmath.ode import calc_fixed_step,euler,rk4
from math import sqrt
import numpy as np


@pytest.mark.parametrize(
    "in_t0,in_n_step,in_t1,in_dt,in_fps,ref_n_step,ref_t1,ref_dt,ref_fps",
    [
        (0,    1,    1, None, None,  1, 1,   1, 1),
        (0,   20, None, None,   10, 20, 2,1/10,10),
        (0, None,    2, None,   10, 20, 2, 1 / 10, 10),

    ]
)
def test_calc_fixed_step(in_t0,in_n_step,in_t1,in_dt,in_fps,ref_n_step,ref_t1,ref_dt,ref_fps):
    """

    :param in_t0:
    :param in_n_step:
    :param in_t1:
    :param in_dt:
    :param in_fps:
    :param ref_n_step:
    :param ref_t1:
    :param ref_dt:
    :param ref_fps:
    :return:
    """
    out_n_step, out_t1, out_dt, out_fps=calc_fixed_step(t0=in_t0,n_step=in_n_step,t1=in_t1,dt=in_dt,fps=in_fps)
    assert out_n_step==ref_n_step
    assert out_t1==ref_t1
    assert out_dt==ref_dt
    assert out_fps==ref_fps


def Fgrav(x, mu):
    x, y, dx, dy = x
    r = sqrt(x ** 2 + y ** 2)
    ddx = mu * x / r ** 3
    ddy = mu * y / r ** 3
    return np.array([dx, dy, ddx, ddy])

def test_euler():
    ref_t1=1
    t1,y1=euler(lambda t,y:Fgrav(y,1), t0=0.0,y0=np.array([1.0,0.0,0.0,1.0]),t1=ref_t1,fps=10)

def test_rk4():
    ref_t1 = 1
    t1, y1 = rk4(lambda t,y:Fgrav(y,1), t0=0.0, y0=np.array([1.0, 0.0, 0.0, 1.0]), t1=ref_t1, fps=10)
