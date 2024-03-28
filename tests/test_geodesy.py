"""
Test geodesy coordinate conversions by running against known DSN coordinates

References:

    [1] "301 Coverage and Geometry." DSN No. 810-005, 301, Rev. M
          Issue Date: September 04, 2020. JPL D-19379; CL#20-3996.

          URL: https://deepspace.jpl.nasa.gov/dsndocs/810-005/301/301M.pdf
    [2] "SPK for DSN Station Locations" comment file
          URL: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/stations/earthstns_fx_201023.cmt

"""
import numpy as np
import pytest as pytest

from kwanmath.geodesy import xyz2lla, lla2xyz
from kwanmath.vector import vcomp

dsn_locations=[
    #station diameter         x             y              z    latd latm  lats{3}  elond elonm elons    h {1}
    # Goldstone, California
    (13,          34, -2351112.659, -4655530.6360, +3660912.728,  35,  14, 49.79131, 243, 12, 19.94761, 1070.444),
    (14,          70, -2353621.420, -4641341.4720, +3677052.318,  35,  25, 33.24312, 243, 6, 37.66244, 1001.390),
    (15,          34, -2353538.958, -4641649.4290, +3676669.984,  35,  25, 18.67179, 243, 6, 46.09762, 973.211),  # {5}
    (24,          34, -2354906.711, -4646840.0950, +3669242.325,  35,  20, 23.61416, 243, 7, 30.74007, 951.499),
    (25,          34, -2355022.014, -4646953.2040, +3669040.567,  35,  20, 15.40306, 243, 7, 28.69246, 959.634),
    (26,          34, -2354890.797, -4647166.3280, +3668871.755,  35,  20,  8.48118, 243, 7, 37.14062, 968.686),
    # Canberra, Australia
    (34,          34, -4461147.093, +2682439.2390, -3674393.133, -35,  23, 54.52383, 148, 58, 55.07191, 692.020),  # {2}
    (35,          34, -4461273.090, +2682568.9250, -3674152.093, -35,  23, 44.86387, 148, 58, 53.24088, 694.897),  # {2}
    (36,          34, -4461168.415, +2682814.6570, -3674083.901, -35,  23, 42.36634, 148, 58, 42.75912, 685.503),  # {2}
    (43,          70, -4460894.917, +2682361.5070, -3674748.152, -35,  24, 8.72724, 148, 58, 52.56231, 688.867),
    (45,          34, -4460935.578, +2682765.6610, -3674380.982, -35,  23, 54.44766, 148, 58, 39.66828, 674.347),  # {5}
    # Madrid, Spain
    (53,          34, +4849339.965, - 360658.2460, +4114747.290,  40,  25, 37.49164, 355, 44, 47.73120, 844.888),  # {4}
    (54,          34, +4849434.488, - 360723.8999, +4114618.835,  40,  25, 32.23805, 355, 44, 45.25141, 837.051),
    (55,          34, +4849525.256, - 360606.0932, +4114495.084,  40,  25, 27.46525, 355, 44, 50.52012, 819.061),
    (56,          34, +4849421.679, - 360549.6590, +4114646.987,  40,  25, 33.47285, 355, 44, 52.58149, 835.746),
    (63,          70, +4849092.518, - 360180.3480, +4115109.251,  40,  25, 52.35510, 355, 45, 7.16924, 864.816),
    (65,          34, +4849339.634, - 360427.6637, +4114750.733,  40,  25, 37.94289, 355, 44, 57.48397, 833.854),
]

@pytest.mark.parametrize(
    "dss,diameter,x,y,z,latd,latm,lats,elond,elonm,elons,h",
    dsn_locations
)
def test_xyz2lla(dss:int,diameter:int,
                 x:float,y:float,z:float,
                 latd:int,latm:int,lats:float,
                 elond:int,elonm:int,elons:float,
                 h:float):
    dss_lla={
        # Geodetic coordinates copied from [2], p15
    # Notes:
    # {1} Geoidal separation must be subtracted from WGS 84 height to get MSL height.
    # {2} Latitude, longitude, and height absolute accuracy estimated to be +/-0.001 sec and +/-3 cm
    #     (0.030 m) (1-sigma)
    # {3} For southern hemisphere antennas deg, min, sec should all be considered negative numbers.
    # {4} Latitude, longitude, and height absolute accuracy estimated to be +/-0.1 sec and +/-3 m
    #     (3-sigma)
    # {5} Decommissioned. For historical reference only.
    }
    xyz=vcomp((x,y,z))
    ref_lat =(np.abs(latd)+ latm/60+ lats/3600)*(-1 if latd<0 else 1)
    ref_elon=       elond +elonm/60+elons/3600
    ref_h=h
    if ref_elon>180.0:
        ref_elon-=360.0
    re=6378137.0 # WGS-84 equatorial radius, copied from [2]
    invf=298.2572236 # Inverse flattening 1/f, copied from [2]
    f=1.0/invf
    rp=(1.0-f)*re
    test_lat,test_lon,test_h=xyz2lla(xyz=xyz,centric=False,deg=True,re=re,rp=rp,array=False)
    if not np.allclose((test_lat,test_lon),(ref_lat,ref_elon)):
        print(f"\ntest lat={test_lat:20.10f} lon={test_lon:20.10f}, h={test_h:10.5f}")
        print(f"ref  lat={ref_lat:20.10f} lon={ref_elon:20.10f}, h={ref_h:10.5f}")
        assert False
    if not np.allclose(test_h,ref_h,atol=0.1):
        print(f"\ntest lat={test_lat:20.10f} lon={test_lon:20.10f}, h={test_h:10.5f}")
        print(f"ref  lat={ref_lat:20.10f} lon={ref_elon:20.10f}, h={ref_h:10.5f}")
        assert False


@pytest.mark.parametrize(
    "dss,diameter,x,y,z,latd,latm,lats,elond,elonm,elons,h",
    dsn_locations
)
def test_lla2xyz(dss:int,diameter:int,
                 x:float,y:float,z:float,
                 latd:int,latm:int,lats:float,
                 elond:int,elonm:int,elons:float,
                 h:float):
    dss_lla={
        # Geodetic coordinates copied from [2], p15
    # Notes:
    # {1} Geoidal separation must be subtracted from WGS 84 height to get MSL height.
    # {2} Latitude, longitude, and height absolute accuracy estimated to be +/-0.001 sec and +/-3 cm
    #     (0.030 m) (1-sigma)
    # {3} For southern hemisphere antennas deg, min, sec should all be considered negative numbers.
    # {4} Latitude, longitude, and height absolute accuracy estimated to be +/-0.1 sec and +/-3 m
    #     (3-sigma)
    # {5} Decommissioned. For historical reference only.
    }
    ref_xyz=vcomp((x,y,z))
    lat =(np.abs(latd)+ latm/60+ lats/3600)*(-1 if latd<0 else 1)
    elon=       elond +elonm/60+elons/3600
    if elon>180.0:
        elon-=360.0
    re=6378137.0 # WGS-84 equatorial radius, copied from [2]
    invf=298.2572236 # Inverse flattening 1/f, copied from [2]
    f=1.0/invf
    rp=(1.0-f)*re
    test_xyz=lla2xyz(lat_deg=lat,lon_deg=elon,alt=h,centric=False,re=re,rp=rp)
    if not np.allclose(test_xyz,ref_xyz):
        print(f"\ntest lat={test_lat:20.10f} lon={test_lon:20.10f}, h={test_h:10.5f}")
        print(f"ref  lat={ref_lat:20.10f} lon={ref_elon:20.10f}, h={ref_h:10.5f}")
        assert False
    assert np.allclose