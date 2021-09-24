"""
Geodesy and gravity calculations
"""
from .vector import vdecomp, vlength, vcomp, rv
import numpy as np

def xyz2llr(sv):
    """
    Calculate spherical coordinates of state
    :param sv: State vector, can be stack
    :return: tuple of (lon,lat,r). Each will be an array iff sv is a stack
    """
    x,y,z=vdecomp(rv(sv))
    r=vlength(rv(sv))
    lat=np.arcsin(z/r)
    lon=np.arctan2(y,x)
    return(lon,lat,r)

def llr2xyz(*,latd,lond,r):
    x=r*np.cos(np.radians(latd))*np.cos(np.radians(lond))
    y=r*np.cos(np.radians(latd))*np.sin(np.radians(lond))
    z=r*np.sin(np.radians(latd))
    return vcomp((x,y,z))

def aJ2(svf,*,j2,gm,re):
    """
    J2 gravity acceleration
    :param svf: State vector in an inertial equatorial frame (frozen frame is designed to meet this requirement)
    :return: J2 acceleration in (distance units implied by rvf)/(time units implied by constants)s**2 in same frame as rvf

    Constants MarsGM, MarsJ2, and MarsR must be pre-loaded and have distance units consistent
    with rvf, and time units consistent with each other.

    Only position part of state is used, but the velocity part *should* have time units consistent
    with the constants. Time units follow those of the constants, completely ignoring those implied
    by the velocity part
    """
    r=vlength(rv(svf))
    coef=-3*j2*gm*re**2/(2*r**5)
    x,y,z=vdecomp(rv(svf))
    j2x=x*(1-5*z**2/r**2)
    j2y=y*(1-5*z**2/r**2)
    j2z=z*(3-5*z**2/r**2)
    return (coef*vcomp((j2x,j2y,j2z)))

def aTwoBody(svi,*,gm):
    """
    Two-body gravity acceleration
    :param rv: Position vector in an inertial frame
    :return: Two-body acceleration in (distance units implied by rv)/s**2
    """
    return -gm*rv(svi)/vlength(rv(svi))**3

