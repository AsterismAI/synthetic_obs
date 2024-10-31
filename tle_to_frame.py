import numpy as np
import pandas as pd
from sgp4.io import twoline2rv
from sgp4.ext import rv2coe
from sgp4.earth_gravity import wgs72
from sgp4.propagation import sgp4 as sgprop
# ---------------- override download ----------------
from astropy.utils import iers
iers.conf.auto_download = False
iers.conf.auto_max_age = None
# ---------------- override download ----------------
import astropy.coordinates
import astropy.time
import astropy.units as u

# -----------------------------------------------------------------------------------------------------
def line_to_tle( L1, L2, earth_grav=None):
    if earth_grav == None: earth_grav = wgs72
    L1 = L1.strip()
    L2 = L2.strip()
    O = twoline2rv( L1, L2, earth_grav )
    # annotate the object
    O.ajd = astropy.time.Time( O.jdsatepoch, format='jd' )  # store epoch as AstroPy time
    O.tleline1 = L1
    O.tleline2 = L2
    return O

# -----------------------------------------------------------------------------------------------------
DEFAULTBAD= ( (np.nan * np.ones(3) ), (np.nan * np.ones(3)) )

# -----------------------------------------------------------------------------------------------------
def load_catalog( filename ):
    with open( filename ) as F: lines = F.readlines()
    L1 = list( filter( lambda X: X[0] == '1', lines))
    L2 = list( filter( lambda X: X[0] == '2', lines))
    tle_obj = [line_to_tle(A,B) for A,B in zip(L1,L2)]
    # de-dupe and take freshest
    outdict = {}
    for t in tle_obj:
        if t.satnum in outdict:
            if t.jdsatepoch > outdict[t.satnum].jdsatepoch: outdict[t.satnum] = t
        else: outdict[t.satnum] = t
    return list( outdict.values() )

# -----------------------------------------------------------------------------------------------------
def TLE2Date( tleobj, adate ):
    min_offset = (adate - tleobj.ajd).to_value(u.min)
    try: return sgprop( tleobj, min_offset )
    except Exception as e: 
        print(e)
        return DEFAULTBAD 

# -----------------------------------------------------------------------------------------------------
def catalogAtDate( listotles, adate ):
    '''
    given a list of MULTIPLE TLE's and a single date, make a frame of
    positions, vels at that date
    '''
    eph = [ TLE2Date( T, adate ) for T in listotles ]
    P,V = zip( *eph )
    return np.hstack( (np.vstack(P), np.vstack(V)) )

# -----------------------------------------------------------------------------------------------------
def tles_to_frame( listotles, adate ):
    eph = catalogAtDate( listotles, adate )
    teme = astropy.coordinates.TEME( x = eph[:,0] * u.km,
                                     y = eph[:,1] * u.km,
                                     z = eph[:,2] * u.km,
                                     v_x = eph[:,3] * u.km/u.s,
                                     v_y = eph[:,4] * u.km/u.s,
                                     v_z = eph[:,5] * u.km/u.s,
                                     obstime = adate,
                                     representation_type = 'cartesian')
    gcrs = teme.transform_to( astropy.coordinates.GCRS( obstime=adate ) )

    gcrs_p = gcrs.cartesian.xyz.to_value(u.km).T
    gcrs_v = gcrs.velocity.d_xyz.to_value(u.km/u.s).T
    N = len(gcrs)
    truth = np.zeros( shape=(N,9) )
    truth[:,0] = [X.satnum for X in listotles] 
    truth[:,1] = adate.jd
    truth[:,2:5] = gcrs_p
    truth[:,5:8] = gcrs_v
    truth[:,8]   = -1
    return { 'TEME'   : teme,
             'GCRS'   : gcrs,
             'truth'  : truth }

# =====================================================================================================
# MAIN
if __name__ == "__main__":
    # TDRS 10
    L1 = '1 27566U 02055A   21130.76031345  .00000095  00000-0  00000-0 0  9998'
    L2 = '2 27566   7.4818  52.0734 0011317 322.5609 316.7791  1.00276956 67549'
    X = line_to_tle( L1, L2 )
    print(X)

