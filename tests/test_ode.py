"""
Foo all the Bars
"""

import pytest
from kwanmath.ode import calc_fixed_step

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
    out_n_step, out_t1, out_dt, out_fps=calc_fixed_step(in_t0,in_n_step,in_t1,in_dt,in_fps)
    assert out_n_step==ref_n_step
    assert out_t1==ref_t1
    assert out_dt==ref_dt
    assert out_fps==ref_fps

def test_euler():
    pass