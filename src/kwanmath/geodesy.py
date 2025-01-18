"""
Geodesy and gravity calculations
"""
from collections import namedtuple

from .vector import vdecomp, vlength, vcomp, rv, vdot
from typing import Union
import numpy as np

def xyz2llr(v,deg=False):
    """
    Calculate spherical coordinates of state
    :param v: column vector, can be stack
    :param deg: If true, return angles in degrees instead of radians
    :return: tuple of (lon,lat,r). Each will be an array iff v is a stack
    """
    x,y,z=vdecomp(v)
    r=vlength(v)
    lat=np.arcsin(z/r)
    lon=np.arctan2(y,x)
    if deg:
        lat=np.rad2deg(lat)
        lon=np.rad2deg(lon)
    return(lon,lat,r)

def llr2xyz(*,lat:float,lon:float=None,heading:float=None,r:float=1.0,deg:bool=True):
    """
    Calculate rectangular coordinates from spherical coordinates

    :param lat: Planetocentric latitude
    :param lon: Planetocentric longitude
                lon=0 is on prime meridian, return vector will be towards +X
                lon=90deg, return vector will be towards +Y
                lon=270deg is same as lon=-90deg, return vector will be towards -Y
    :param heading: heading east of true north, only used if lon is not passed. With heading:
                    heading=0 is true north, return vector will be towards +Y
                    heading=90deg is true east, return vector will be towards +X
    :param r: Radius
    :param deg: True if latitude and longitude are in degrees
    :return: Rectangular coordinates

    Only one of lon= or heading= should be passed. Parameter lon= has priority and is used
    if both are passed.

    Note: Uses array operations. All inputs may be array, but must be broadcastable against each other.
    """
    if lon is None:
        lon=(90 if deg else np.pi/2)-heading
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


def lla2xyz(*,centric:bool,lat_deg:Union[float,np.array]=None, lat_rad:float=None,
            lon_deg:float=None, lon_rad:float=None,
            alt:float=None, re:float=None, rp:float=None, east:bool=True):
    """
    Calculate rectangular coordinates from geodetic coordinates

    :param lat_deg: Latitude in degrees, may be an array
    :param lon: Longitude in degrees, may be an array
    :param alt: Altitude above ellipsoid, may be an array
    :param re: Equatorial radius of ellipsoid
    :param rp: Polar radius of ellipsoid
    :return: stack of vectors

    """
    if lat_rad is None:
        lat_rad=np.deg2rad(lat_deg)
    if lon_rad is None:
        lon_rad=np.deg2rad(lon_deg)
    if centric:
        if alt is None:
            alt = 0
        a = np.tan(lat_rad) ** 2 + rp ** 2 / re ** 2
        c = -rp ** 2
        r_ssc = np.sqrt(a * -c) / a
        z_ssc = np.sqrt(rp ** 2 * (1 - r_ssc ** 2 / re ** 2))
        R_ssc = np.sqrt(r_ssc ** 2 + z_ssc ** 2)
        R = R_ssc + alt
        z = R * np.sin(lat_rad)
        r = R * np.cos(lat_rad)
    else:
        # From "Geodetic to Cartesian" of Borkowski paper
        # http://www.astro.uni.torun.pl/~kb/Papers/geod/Geod-BG.htm
        psi = np.arctan(rp * np.tan(lat_rad) / re)  # Reduced latitude
        r = re * np.cos(psi)
        z = rp * np.sin(psi)
        if alt is not None:
            r += alt * np.cos(lat_rad)  # Distance from rotation axis
            z += alt * np.sin(lat_rad)  # Distance from equatorial plane
    x = r * np.cos(lon_rad)
    y = r * np.sin(lon_rad) * (1 if east else -1)
    return vcomp((x, y, z))


def xyz2lla(*,centric:bool, deg:bool, xyz:np.array, re:float, rp:float, east:bool=True,array:bool=False):
    """
    Transform rectangular coordinates to geodetic coordinates.

    :param xyz: Stack of rectangular coordinates of the point in a frame
                aligned to the equator and poles of the spheroid in the same
                units as the equatorial radius
    :type  xyz:  Numpy array with second-to-last dimension of size 3 (stack of 3-element vectors)
    :param re:   equatorial radius of spheroid
    :type  re:    float
    :param rp:   Polar radius os spheroid
    :type  rp:    float
    :param east: If true, return values using the positive East convention,
                          otherwise the positive West convention
    :param centric: If true, use planetocentric rather than planetodetic latitude and altitude. Planetocentric
                    latitude is obvious. Altitude is the difference between the position vector length and
                    the planetocentric radius of the ellipsoid at this planetocentric latitude IE the length
                    of the line with a slope angle equal to the given latitude, through the center, from the center
                    to the ellipsoid.
    :return: * **lat** (Numpy stack of scalars, same shape as input stack of vectors) - Latitude in degrees
             * **lon** (Numpy stack of scalars, same shape as input stack of vectors) - Longitude in degrees
             * **alt** (Numpy stack of scalars, same shape as input stack of vectors) -
                           Distance from surface of spheroid in same units as equatorial radius
    :rtype: namedtuple

    """
    # Transform the input arguments from a stack of vectors to 3 arrays of coordinates
    x,y,z=vdecomp(xyz,array=array)
    # Nothing special about longitude, except that this always gives longitude
    # from -180deg to +180deg. Positive is EAST if east= parameter is True (default).
    lon = np.arctan2(y, x) * (1 if east else -1)

    if centric:
        # We *don't* scale to units of equatorial radius in this branch,
        # to follow the original derivation (Planetocentric.ipynb)

        # Length of projection of vector to equatorial plane
        r = np.sqrt(x ** 2 + y ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            lat = np.rad2deg(np.arctan(z / r))
            a = z ** 2 / r ** 2 + rp ** 2 / re ** 2
            c = -rp ** 2
            r_ssc = np.sqrt(a * -c) / a
            z_ssc = np.sqrt(rp ** 2 * (1 - r_ssc ** 2 / re ** 2))
        R_ssc = np.sqrt(r_ssc ** 2 + z_ssc ** 2)
        R = np.sqrt(r ** 2 + z ** 2)
        alt = R - R_ssc
        result = namedtuple('xyz2lla', ['lat', 'lon', 'alt'])
        return result(lat, lon, alt)

    # This is done following the Borkowski algorithm,
    # translated from my implementation in C/C++

    # scale to units of equatorial radius
    x = x / re
    y = y / re
    z = z / re

    # Length of projection of vector to equatorial plane
    r = np.sqrt(x ** 2 + y ** 2)

    # With that out of the way, re should not appear below except to re-scale
    # the altitude to the original units

    # Ellipsoid equatorial radius is 1 unit
    # Ellipsoid polar radius, units. Note that b is an array. Each element has
    # the same magnitude but the sign of the z component of the vector
    with np.errstate(divide='ignore', invalid='ignore'):
        b = z / np.absolute(z) * (rp / re)
    lat = np.ndarray(b.shape)
    alt = np.ndarray(b.shape)

    # Handle the polar (r==0) case. Unlikely to happen exactly
    on_pole = (r == 0)
    off_pole = np.logical_not(on_pole)
    lat[on_pole] = z[on_pole] / np.absolute(z[on_pole]) * np.pi / 2;
    alt[on_pole] = np.absolute(z[on_pole]) - np.absolute(b[on_pole]);
    # Handle the equatorial (z==0) case. Unlikely to happen exactly
    on_equ = (z == 0)
    off_equ = np.logical_not(on_equ)
    lat[on_equ] = 0;
    alt[on_equ] = r[on_equ] - 1;
    # Handle the nominal case, off-polar AND off-equatorial
    nominal = np.logical_and(off_pole, off_equ)
    E = ((z[nominal] + b[nominal]) * b[nominal] - 1.0) / r[nominal];
    F = ((z[nominal] - b[nominal]) * b[nominal] + 1.0) / r[nominal];
    P = 4.0 * (E * F + 1.0) / 3.0;
    Q = (E * E - F * F) * 2.0;
    D = P * P * P + Q * Q;
    Dge0 = (D >= 0)
    Dlt0 = np.logical_not(Dge0)
    v = np.ndarray(D.shape)
    v[Dge0] = np.cbrt(np.sqrt(D[Dge0]) - Q[Dge0]) - np.cbrt(np.sqrt(D[Dge0]) + Q[Dge0]);
    v[Dlt0] = 2.0 * np.sqrt(-P[Dlt0]) * np.cos(np.arccos(Q[Dlt0] / P[Dlt0] / np.sqrt(-P[Dlt0])) / 3.0);
    G = (E + np.sqrt(E * E + v)) / 2.0;
    t = np.sqrt(G * G + (F - v * G) / (G + G - E)) - G;
    lat[nominal] = np.arctan((1.0 - t * t) / (2 * b[nominal] * t));
    alt[nominal] = (r[nominal] - t) * np.cos(lat[nominal]) + (z[nominal] - b[nominal]) * np.sin(lat[nominal]);
    result = namedtuple('xyz2lla', ['lat', 'lon', 'alt'])
    return result(np.rad2deg(lat) if deg else lat, np.rad2deg(lon) if deg else lon, alt * re)


def ray_sphere_intersect(r0, v, re):
    """
    Calculate the point of intersection between a given ray r(t)=r0+v*t and a sphere dot(r,r)=re**2

    :param numpy vector r0: initial point of ray
    :param numpy vector v:  direction of ray. If this is a unit vector, then the units of t returned will be
                            the same as the units of r0
    :param float re: radius of sphere
    :return: tuple, first element is ray parameter, second is intersect point.

    Parametric equation for a ray
      r(t)=r0+v*t
    Equation for a sphere of radius r_m
      dot(r,r)=re**2
    solve simultaneous equations
      dot(r0+v*t,r0+v*t)=re**2
      (rx+vx*t)**2+                           #expand dot product to components
      (ry+vy*t)**2+
      (rz+vz*t)**2=r_m**2
      (r_+v_*t)**2=r_**2+2*r_*v_*t+v_**2*t**2 #square each term
      rx**2+2*rx*vx*t+vx**2*t**2+             #substitute terms
      ry**2+2*ry*vy*t+vy**2*t**2+
      rz**2+2*rz*vz*t+vz**2*t**2=r_m**2
      t**2*(vx**2  +vy**2  +vz**2  )+         #gather into coefficients of P(t)
      t**1*(2*rx*vx+2*ry*vy+2*rz*vz)+
      t**0*(rx**2  +ry**2  +rz**2  )=re**2
      t**2*dot(v,v)+                          #recognize dot products
      t**1*2*dot(r0,v)+
      t**0*dot(r,r)=re**2
    So, the quadratic coefficients are:
      * A=dot(v,v)
      * B=2*dot(r0,v)
      * C=dot(r0,r0)-re**2
    """
    A = vdot(v, v)
    B = 2 * vdot(r0, v)
    C = vdot(r0, r0) - re ** 2
    D = B ** 2 - 4 * A * C
    # Since A is positive, it is always the case that using the negative sign will give the
    # lower root, which is what we want. If this root is negative, then the spacecraft is
    # inside the sphere or the sphere is behind the spacecraft.
    t = (-B - np.sqrt(D)) / 2 * A
    # Finish using the ray equation to find the coordinates of p1 from the spacecraft pos/vel
    return (t, r0 + v * t)






