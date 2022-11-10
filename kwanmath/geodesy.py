"""
Geodesy and gravity calculations
"""
from .vector import vdecomp, vlength, vcomp, rv
import numpy as np

def xyz2llr(sv,deg=False):
    """
    Calculate spherical coordinates of state
    :param sv: State vector, can be stack
    :param deg: If true, return angles in degrees instead of radians
    :return: tuple of (lon,lat,r). Each will be an array iff sv is a stack
    """
    x,y,z=vdecomp(rv(sv))
    r=vlength(rv(sv))
    lat=np.arcsin(z/r)
    lon=np.arctan2(y,x)
    if deg:
        lat=np.rad2deg(lat)
        lon=np.rad2deg(lon)
    return(lon,lat,r)

def llr2xyz(*,lat,lon,r=1.0,deg=True):
    """
    Calculate rectangular coordinates from spherical coordinates

    :param lat: Planetocentric latitude
    :param lon: Planetocentric longitude
    :param r: Radius
    :param deg: True if latitude and longitude are in degrees
    :return: Rectangular coordinates

    Note: Uses array operations. All inputs may be array, but must be broadcastable against each other.
    """
    if deg:
        lat=np.deg2rad(lat)
        lon=np.deg2rad(lon)
    clat=np.cos(lat)
    clon=np.cos(lon)
    slat=np.sin(lat)
    slon=np.sin(lon)
    x=r*clat*clon
    y=r*clat*slon
    z=r*slat
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

