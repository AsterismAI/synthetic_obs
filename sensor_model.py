#!/usr/bin/env python
# coding: utf-8

# # Sensor Model (NEW)
#
# Kerry N. Wood (kerry.wood@jhuapl.edu)
#
# June 10, 2021
#
# - this is a new version of the sensor model designed to be faster (lots of vector math)
# - it NO LONGER supports multiple dates (obstimes) per frame... just ONE
#
# August 27, 2021
# - removing all pandas references 
#
# January 04, 2021
# - started the refactor

# ---------------- override download ----------------
from astropy.utils import iers
iers.conf.auto_download = False
iers.conf.auto_max_age = None

# ---------------- imports ----------------
import astropy
import astropy.units as u
import time
#from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import requests
import zipfile
import io
import os
import sys

from custom_logger import logger

# -------------------------------------  PATH HACK -------------------------------------
common = os.path.join( os.path.abspath('../'), 'common' )
if common in sys.path: logger.info('found path already set {}'.format(common))
else: sys.path.append( common )

# ------------------------------------- LOCALS -------------------------------------
import solar_calcs

# ----------------------------------------------------------------------------------------------------- 
# !!!! (KNW) : if you add a field here, you must add the appropriate outputter in buildOutput below
SENSTYPE = { 0 : ['RANGERATE'],
        1  : ['AZ','EL'],
        2  : ['AZ','EL','RANGE'],
        3  : ['AZ','EL','RANGE','RANGERATE'],
        4  : ['AZ','EL','RANGE','RANGERATE','ELRATE','AZRATE','RANGEACCEL'],
        5  : ['RA','DEC'],
        6  : ['RANGE'],
        8  : ['AZ','EL'],
        9  : ['RA','DEC'] }
        #10 : ['POSX','POSY','POSZ','VELX','VELY','VELZ'],
        #11 : ['POSX','POSY','POSZ'],
        #18 : ['AZ','EL','RANGE'],
        #19 : ['RA','DEC','RANGE'],


# =====================================================================================================
def buildOutput( sens, gcrs_input, truth_statevec,
                       sensor_gcrs, obs_gcrs,
                       viewable,
                       altaz=None, 
                       solar_phase=None, solar_angle=None ):
    '''
    in every case we have common data:
        gcrs_input the : "truth frame" converted to an AstroPy GCRS frame (object locations with no set observer)
        truth_statevec : a numpy matrix that is <scc>,<jd>,<gcrsx>,<gcrsy>,<gcrsz>,<gcrsdx>,<gcrsdy>,<gcrsdz>,<sun>
        sensor_gcrs    : the location of the sensor in GCRS (for vector math to celestial)
        obs_gcrs       : the truth frame as seen from the sensor (in GCRS, for RA/DEC calcs)
        viewable       : bool array indicating whether objects are viewable from the sensor
        altaz [optional] : the altaz frame 
    '''
    #solar_angle, solar_phase = solar_calcs.solar_calcs( gcrs_input, sensor_gcrs )
    # KNW: AstroPy requires you do frame transforms to get look vectors.  Here, you'll see lots of frames: some with
    # raw positions, some with calculated looks.  I'm keeping all of them for now, could probably be reduced later.
    rv =  { 'name'        : sens.name,
            'type'        : sens.type,
            'input_gcrs'  : gcrs_input,  # raw GCRS position of all objects (without calculating views from sensor)
            'sensor_gcrs' : sensor_gcrs, # raw GCRS position of the sensor (again, no computation of looks)
            'obs_gcrs'    : obs_gcrs,    # GCRS frame that computes looks to `input_gcrs` from `sensor_gcrs`
            'inFOR'       : viewable,    # viewable flag
            'solar_phase' : solar_phase, # solar phase
            'solar_angle' : solar_angle, # solar angle
            'noised'      : {} ,
            'raw_noise'   : {}}
    if sens.type != 8 and sens.type != 9 : rv['earth_loc'] = sens.get_lla()
    if truth_statevec.shape[1] == 9      : rv['sunlit'] = truth_statevec[:,8]

    rv['scc']              = truth_statevec[:,0]
    N                      = len(gcrs_input)
    if altaz : rv['altaz'] = altaz
    # /////////////////////////////////////////////////////////////////////////////////////////////////////
    # !!!!!!!!!!!!!!!!!!!!!  you must map the physics to the outputs here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    if 'AZ' in sens.fields         : rv['AZ']         = altaz.az.to_value(u.deg)
    if 'EL' in sens.fields         : rv['EL']         = altaz.alt.to_value(u.deg)
    if 'RA' in sens.fields         : rv['RA']         = obs_gcrs.ra.to_value(u.deg)
    if 'DEC' in sens.fields        : rv['DEC']        = obs_gcrs.dec.to_value(u.deg)
    if 'RANGE' in sens.fields      : rv['RANGE']      = altaz.distance.to_value(u.km)
    if 'RANGERATE' in sens.fields  : rv['RANGERATE']  = altaz.radial_velocity.to_value(u.km/u.s)
    if 'AZRATE' in sens.fields     : rv['AZRATE']     = altaz.pm_alt.to_value(u.deg/u.s)
    if 'ELRATE' in sens.fields     : rv['ELRATE']     = altaz.pm_az_cosalt.to_value(u.deg/u.s)
    if 'RANGEACCEL' in sens.fields : rv['RANGEACCEL'] = np.zeros( N )  # TODO: we need to handle RANGEACCEL

    # -----------------------------------------  APPLY NOISE -----------------------------------------
    for fld in sens.fields:
        rv['raw_noise'][fld] = sens.getNoise( fld, N )
        rv['noised'][fld]    = rv[fld] + rv['raw_noise'][fld]
    return rv


# =====================================================================================================
class sensor_base:
    def __init__( self, data = None, name=None ):
        self.data = data 
        if self.data is None: return

        self.viewbox  = []
        self.astro_standards_number = self.data.get('SENSOR_NUMBER',None)
        self.solar_exclusion        = self.data.get('SOLAR_EXCLUSION_ANGLE',None)
        self.lunar_exclusion        = self.data.get('LUNAR_EXCLUSION_ANGLE',None)
        self.type                   = self.data.get('OBSERVATION_TYPE', -1 )
        self.name                   = self.data.get('DESCRIPTION', "NONAME") 
        self.require_illuminated    = False
        self.sigmas                 = {}

        # space based can come from here too
        self.FOV                    = self.data.get('FOV', None)
        self.NORAD                  = self.data.get('NORAD', None)

        # make sure that we have a valid sensor type
        assert self.type in SENSTYPE
        self.fields                 = SENSTYPE[ self.type ]

        # build the error fields
        for fld in self.fields:
            self.sigmas[ fld ] = {}
            self.sigmas[ fld ]['bias']  = self.data.get( '{}BIAS'.format( fld )  , 0.0 )
            self.sigmas[ fld ]['sigma'] = self.data.get( '{}SIGMA'.format( fld ) , 0.0 )

        # peel out a few key fields
        # TODO (KNW) : these flags make no sense, we need a better way to dictate:
        # - if a sensor needs to be in darkness
        # - if a sensor needs the target illuminated
        # - all the combinations of exclusions
        if 'OPTICAL_TYPE' in self.data and self.data['OPTICAL_TYPE'] > 0: 
            self.twilight            = self.data.get("TWILIGHT_ANGLE",None)

        # get the AstroStandards number
        self.astro_standards_number = self.data.get('SENSOR_NUMBER',None)

        # get the NORAD number
        self.NORAD                  = self.data.get('NORAD',None)

    def getBias( self, field ): 
        try: return self.sigmas[ field ]['bias']
        except: return None

    def getSigma( self, field ): 
        try: return self.sigmas[ field ]['sigma']
        except: return None

    def getNoise( self, field, N ):
        return np.random.normal( self.getBias(field), self.getSigma(field), N )

# ----------------------------------------------------------------------------------------------------- 
class sensor_ground_site( sensor_base ):
    '''
    build synthetic obs based on AstroStandard (or AstroStandard-like) data
    note that we pass around a specific time, even though it might also be in the frame...  that's to prevent confusion with possible multiples
    '''
    def __init__( self, sensordict, name='NA' ):
        super().__init__( sensordict )
        self.build_earthloc()
        self.buildFOR()

    def get_lla( self ):
        lla = self.earth_loc.geodetic
        try:
            lla = self.earth_loc.geodetic
            return {'lat' : lla.lat.to_value(u.deg),
                    'lon' : lla.lon.to_value(u.deg),
                    'alt' : lla.height.to_value(u.km) }
        except Exception as e:
            return {'lat':np.nan, 'lon':np.nan,'alt':np.nan }

    def get_ecr( self ):
        try:
            #ecr = self.earth_loc.geocentric
            #return [ A.to_value(u.km) for A in ecr ]
            return self.earth_loc.geocentric.to_value(u.km)
        except Exception as e:
            return np.nan * np.ones(3)

    def build_earthloc(self):
        self.lat = self.data['LATITUDE']
        self.lon = self.data['LONGITUDE']
        self.alt = self.data['ALTITUDE']
        self.earth_loc= astropy.coordinates.EarthLocation.from_geodetic( 
                                                lat        = self.lat * u.deg,
                                                lon        = self.lon * u.deg,
                                                height     = self.alt * u.km,
                                                ellipsoid  = 'WGS84' )

    def llaJSON( self ):
        return json.dumps( self.get_lla() )

    def buildFOR( self ):
        '''
        set up the field-of-regard using clock angles
        '''
        az1 = self.data['AZIMUTH_LEFT_1']
        az2 = self.data['AZIMUTH_RIGHT_1']
        az3 = self.data['AZIMUTH_LEFT_2']
        az4 = self.data['AZIMUTH_RIGHT_2']
        el1 = self.data['ELEVATION_LOW_1']
        el2 = self.data['ELEVATION_HIGH_1']
        el3 = self.data['ELEVATION_LOW_2']
        el4 = self.data['ELEVATION_HIGH_2']
        self.viewbox = [ (az1,az2,el1,el2) ] 
        # some sensors have two fields of regard
        if az3 != az4: self.viewbox.append( (az3,az4,el3,el4 ) )
        # add in range
        self.range = (self.data['MIN_RANGE'], self.data['MAX_RANGE'] )
        self.solar_exclusion = self.data['SOLAR_EXCLUSION_ANGLE']
        self.lunar_exclusion = self.data['LUNAR_EXCLUSION_ANGLE']

    def locstr( self ): return " ".join(["{:010.3f}".format(X) for X in list(self.get_lla().values()) ] )
    def __repr__( self ): return "{:>25}(type: {:1d}) {}".format( self.name, self.type, self.locstr() )

    def azel_of_frame( self, gcrs_frame ):
        '''
        we always need the AltAz frame to check view boxes (TODO: except space-based.. but we'll figure that out)
        given known locations of objects (and a single time),
        '''
        altaz = astropy.coordinates.AltAz( location=self.earth_loc, obstime=gcrs_frame.obstime )
        local = gcrs_frame.transform_to( altaz )
        return local

    def site_azel_to_gcrs( self, astropy_time, az, el ):
        '''
        given an astropy time and az el, return a GCRS look vector and current sensor position
        '''
        sengcrs = self.getSensorLocGCRS( now )
        look = astropy.coordinates.AltAz( location = self.earth_loc,
                                          az       = az*u.deg,
                                          alt      = el*u.deg,
                                          distance = 1e15*u.km,
                                          obstime  = now)
        look_gcrs = look.transform_to( astropy.coordinates.GCRS( obstime   = now,
                                                                 obsgeoloc = sengcrs.cartesian.xyz,
                                                                 obsgeovel = sengcrs.velocity.d_xyz ) )
        lgcrs = look_gcrs.cartesian.xyz.to_value(u.km)
        sgcrs = Z.cartesian.xyz.to_value(u.km)
        diff = lgcrs - sgcrs
        return diff / np.linalg.norm(diff)

    def checkView( self, gcrs, azel, vbox ):
        ''' clock angle check to see if field-of-regard is satisfied '''
        alt   = azel.alt.to_value(u.deg)
        az    = azel.az.to_value(u.deg)
        L,R,B,T = vbox # left, right, bottom, top
        if L < R : running = (az >= L) & (az <= R )
        if L > R : running = (az >= R) & (az <= L )
        running = (alt >=B) & (alt <= T ) & running
        return running

    def checkRange( self, gcrs, azel, rng ):
        '''
        rng = (range_low, range_high)
        '''
        lR, hR = rng
        dist  = azel.distance.to_value(u.km)
        return (dist >= lR) & (dist <= hR)

    def checkFOV( self, gcrs, azel ):
        # KNW (TODO) : we should do a sun check here ( twilight / exclusion )
        N = len(azel)
        goodRange = self.checkRange( gcrs, azel, self.range)
        # some sensors have more than one region; hence the OR conditions
        goodViewBox = np.array( [False for _ in range(N)])
        for B in self.viewbox: goodViewBox = goodViewBox | self.checkView( gcrs, azel, B )
        return goodRange & goodViewBox

    def getSensorLocGCRS( self, obstime ): return self.earth_loc.get_gcrs( obstime ) # OLD version / slow

    def getSensorLocITRS( self, obstime ): return self.earth_loc.get_itrs().cartesian.xyz.to_value(u.km)

    def getObData( self, gcrs_input, truth_statevec, justViewable=False ):
        sensorgcrs = self.getSensorLocGCRS( gcrs_input.obstime )
        # ---------------------  SUN CHECK ---------------------
        suncheck    = np.full( len(gcrs_input), True, dtype=bool )
        solar_phase = np.full( len(gcrs_input), np.nan, dtype=bool )
        solar_angle = np.full( len(gcrs_input), np.nan, dtype=bool )
        sun_gcrs    = astropy.coordinates.get_sun( gcrs_input.obstime )
        # check twilight
        if self.twilight:
            sunaltaz = self.azel_of_frame( sun_gcrs )
            if sunaltaz.alt.to_value(u.deg) > self.twilight:  return None # <--- return None if sun is up
        # check for solar exclusion
        solar_angle, solar_phase, sun_gcrs = solar_calcs.solar_calcs( gcrs_input, sensorgcrs, sun_gcrs )
        if self.solar_exclusion: suncheck = solar_angle > self.solar_exclusion
        else: suncheck = np.full( len(gcrs_input), True, dtype=bool )
        # ---------------------  FOR CHECK ---------------------
        # look-box check
        altaz = self.azel_of_frame( gcrs_input )
        inFOR = self.checkFOV( gcrs_input, altaz )
        viewable = suncheck & inFOR
        # ---------------------  OUTPUT ---------------------
        if justViewable:
            if np.sum(viewable) == 0: return None
            solar_phase = solar_phase[viewable]
            solar_angle = solar_angle[viewable]
            altaz = altaz[viewable]
            gcrs_input = gcrs_input[ viewable ]
            truth_statevec = truth_statevec[ viewable, : ]
            #viewable = np.ones( pd_frame.shape[0], dtype=np.bool ) * True
            viewable = viewable[ viewable ]
        obsframe = astropy.coordinates.GCRS( obstime   = gcrs_input.obstime,
                                             obsgeoloc = sensorgcrs.cartesian.xyz,
                                             obsgeovel = sensorgcrs.velocity.d_xyz)
        newframe = gcrs_input.transform_to( obsframe )
        return buildOutput( self, gcrs_input, truth_statevec, sensorgcrs, newframe, viewable, 
                                altaz=altaz, solar_phase=solar_phase, solar_angle=solar_angle ) 

    # -------------------------------  make obs -------------------------------
    def makeObs( self, gcrs_in_astropy, truth_statevec, justViewable=False):
        return self.getObData( gcrs_in_astropy, truth_statevec, justViewable=justViewable )

    def make_SOAP( self, color='Blue', shape='PLACEMARK', label=True ):
        lat,lon,alt = list(self.get_lla().values())
        if 'astro_standards_number' in dir(self): 
            rs = 'DEFINE PLATFORM ECR_FIXED  "{}"'.format(self.astro_standards_number)
        else : rs = 'DEFINE PLATFORM ECR_FIXED  "{}"'.format(self.name)
        rs += '\n\tSTATE  {} {} {}'.format( lat, lon, alt )
        rs += '\n\tSHAPE {}'.format( shape )
        rs += '\n\tCOLOR {}'.format( color )
        rs += '\n\t ICON ON'
        rs += '\n\t ABOVE_TERR OFF'
        if label: rs += '\n\tLABEL ON'
        else: rs+= '\n\tLABEL OFF'
        return rs

# ------------------------------------------------ BASE ------------------------------------------------
class sensor_space_based( sensor_base ):
    '''
    build synthetic obs based on space-based ; MUST be given a GCRS position and velocity
    space-based does not rely on az/el checks and will work in the GCRS frame

    note that we pass around a specific time, even though it might also be in the frame...  that's to prevent confusion with possible multiples

    NOTE: < -----------------------------------------
    - this DOES NOT work like ground based sensors for multiple times (you can't broadcast across times)
    '''
    def __init__( self, data=None, NORAD=None, name=None, type=9, FOV=2.0, solar_exclusion=30., sensor_number=None ):
        if data is None    : data = {}
        if NORAD           : data['NORAD'] = NORAD
        else               : data['NORAD'] = 999999
        if name            : data['DESCRIPTION'] = name
        else:                data['DESCRIPTION'] = 'SPACEBASED:{:06d}'.format( data['NORAD'] )
        if type            : data['OBSERVATION_TYPE'] = type
        if solar_exclusion : data['SOLAR_EXCLUSION_ANGLE'] = solar_exclusion
        if sensor_number   : data['SENSOR_NUMBER'] = sensor_number
        # now, init with that fake dict
        if data            : super().__init__( data )
        if FOV             : self.FOV   = FOV
        if solar_exclusion : self.solar_exclusion = solar_exclusion

    def checkView( self, gcrs, sensor_gcrs_coor, pointing_vec=None ):
        # KNW (TODO) : we should do a sun check here ( twilight / exclusion )
        N = gcrs.shape[0]
        pointing_vec = (pointing_vec / np.linalg.norm( pointing_vec )) * np.ones( (N,3) )
        gcrs_pos = gcrs.cartesian.xyz.to_value(u.km).T
        sen_gcrs = sensor_gcrs_coor.cartesian.xyz.to_value(u.km).T * np.ones((N,3))
        # pointing vecs
        sat_to_sen = gcrs_pos - sen_gcrs
        sat_to_sen /= np.linalg.norm( sat_to_sen, axis=1 )[:,np.newaxis]
        if pointing_vec is not None : 
            angle = np.degrees( np.arccos( np.einsum('ij,ij->i',sat_to_sen,pointing_vec) ) )
            return angle <= (self.FOV/2.)
        return [ True for _ in len( sat_to_sen ) ]


    def getSpaceBased( self, gcrs_input, sensor_gcrs_coor, truth_statevec, look_vector, justViewable = False ):
            '''
            almost all data is made for each ob, so why not just build it en masse, and then parse?
            gcrs_input        : a frame containing all viewable objects in an AstroPy GCRS frame
            sensor_gcrs_coor  : the location of the sensor, this should be pulled from the frame
            look_vector       : the look vector (from sensor_gcrs_coor)

            if senGCRSpos/vel are input, use those as your sensor location (for space-based)
            if they're not, fall back on the earthlocation provided in setup
            '''
            # ------------------------------- FOR CHECK -------------------------------
            # at this point, we need to see what is visible
            thisjd = gcrs_input.obstime
            viewable = self.checkView( gcrs_input, sensor_gcrs_coor, look_vector )
            # ------------------------------- SUN CHECK -------------------------------
            solar_angle, solar_phase, sun_gcrs = solar_calcs.solar_calcs( gcrs_input, sensor_gcrs_coor )
            if self.solar_exclusion: suncheck = solar_angle > self.solar_exclusion
            else: suncheck = np.full( len(gcrs_input), True, dtype=bool )
            viewable = suncheck & viewable
            # make sure you don't task on yourself
            #viewable[ truth_statevec[:,0] == self.NORAD ] = False

            if justViewable:
                if np.sum(viewable) == 0: return None
                gcrs_input     = gcrs_input[ viewable ]
                truth_statevec = truth_statevec[ viewable, : ]
                solar_angle    = solar_angle[ viewable ]
                solar_phase    = solar_phase[ viewable ]
                viewable       = viewable[viewable] # MUST BE LAST
            obsframe = astropy.coordinates.GCRS( obstime   = gcrs_input.obstime,
                                                 obsgeoloc = sensor_gcrs_coor.cartesian.xyz,
                                                 obsgeovel = sensor_gcrs_coor.velocity.d_xyz)
            newframe = gcrs_input.transform_to( obsframe )

            return buildOutput( self, gcrs_input, truth_statevec, sensor_gcrs_coor, newframe, viewable ,
                    solar_phase=solar_phase, solar_angle=solar_angle)

    def findMeInFrame( self, gcrs_in_astropy, truth_statevec) :
        #myidx = np.where( pd_dataframe['scc'] == self.NORAD )[0][0]
        myidx = np.where( truth_statevec[:,0] == self.NORAD )[0][0]
        return gcrs_in_astropy[myidx]

    def makeObs( self, gcrs_in_astropy, truth_statevec, look_vector=None, justViewable=True ):
        '''
        for space-based, we need the STATE MANAGER to tell us where the sensor is and tell us the velocity
        this is unlike the ground-based AstroStandard stuff, that has a fixed location
        '''
        try: my_gcrs = self.findMeInFrame( gcrs_in_astropy, truth_statevec) 
        except Exception as e:
            logger.error('could not find {} in frame, cannot do space based, error {}'.format( self.NORAD, e ))
            return None
        if look_vector == None:
            look_vector = my_gcrs.velocity.d_xyz.to_value(u.km/u.s)
            look_vector = look_vector / np.linalg.norm( look_vector )
        return self.getSpaceBased( gcrs_in_astropy, my_gcrs, truth_statevec, look_vector, justViewable=justViewable )

    #def __repr__( self ): return "{} SCC: {}  type: {}".format(self.name, self.NORAD, self.type)
    def __repr__( self ): return "{:>25}(type: {:1d}) SCC: {:05d}".format( self.name, self.type, self.NORAD ) 

    def make_SOAP( self, color='Blue', shape='PLACEMARK', label=True ):
        return ''
#         lat,lon,alt = self.earth_loc_LLA
#         rs = 'DEFINE PLATFORM ECR_FIXED  "{}"'.format(self.name)
#         rs += '\n\tSTATE  {} {} {}'.format( lat, lon, alt )
#         rs += '\n\tSHAPE {}'.format( shape )
#         rs += '\n\tCOLOR {}'.format( color )
#         rs += '\n\t ICON ON'
#         rs += '\n\t ABOVE_TERR OFF'
#         if label: rs += '\n\tLABEL ON'
#         else: rs+= '\n\tLABEL OFF'
#         return rs


# ------------------------------------------------ ORS ------------------------------------------------
class sensor_ors5( sensor_space_based ):
    '''
    build synthetic obs based on space-based
    MUST be given a GCRS position and velocity

    note that we pass around a specific time, even though it might also be in the frame...  that's to prevent confusion with possible multiples
    '''
    def __init__( self, name=None, angle_offset=22 ):
        if name is None: name ='ORS5'
        super().__init__(NORAD=42921, name=name, FOV=6.0, type=9 )
        self.lv = None
        self.type = 9
        rot = np.radians( angle_offset )
        self.obs_rotation = np.array([[np.cos(rot), -np.sin(rot), 0. ],
                                    [np.sin(rot), np.cos(rot), 0. ],
                                    [0, 0, 1] ] )
        self.astro_standards_number = 511

    def getPointing( self, sensor_gcrs_coor ):
        '''
        ORS view is always an angle_offset from velocity
        '''
        sen_gcrs = sensor_gcrs_coor.cartesian.xyz.to_value(u.km).T
        sen_vel  = sensor_gcrs_coor.velocity.d_xyz.to_value(u.km/u.s).T
        # get the look vector
        point_vec = np.dot( sen_vel, self.obs_rotation )
        point_vec = point_vec / np.linalg.norm( point_vec )
        return point_vec

    def makeObs( self, gcrs_in_astropy, truth_statevec, look_vector=None, justViewable=False ):
        try: my_gcrs = self.findMeInFrame( gcrs_in_astropy, truth_statevec )
        except Exception as e:
            logger.error('could not find {} in frame, cannot do space based, error {}'.format( self.NORAD, e ))
            return None
        # KNW : this is a hacky way to get some "last sensor task status" out of the object
        self.lv   = self.getPointing( my_gcrs )
        self.gcrs = my_gcrs
        return self.getSpaceBased( gcrs_in_astropy, my_gcrs, truth_statevec, self.lv, justViewable=justViewable)


# .....................................................................................................
def test_ors5():
    ors5 = sensor_ors5()
    print( ors5.makeObs( *testframes[0] ) )
    return ors5

# =====================================================================================================
if __name__ == "__main__":
    import os
    import sys
    from tle_to_frame import tles_to_frame, line_to_tle

    # ------------------  LOAD IN SENSORS -----------------------
    as_sensors = pd.read_csv( './data/obfuscated_sensors.csv', index_col=False ).to_dict('records')
    as_sensors = [sensor_ground_site(S) for S in as_sensors]
    print('Loaded {:04d} sensors'.format( len(as_sensors) ) )

    # ------------------  space based are different! -------------------
    # here, we'll pass in the state based on the universal state and gin up obs accordingly
    ors = sensor_ors5(name='ORS5')
    as_sensors.append( ors )

    # ------------------------------- LOAD TLE AND TEST -------------------------------
    L1 = '1 27566U 02055A   21130.76031345  .00000095  00000-0  00000-0 0  9998'
    L2 = '2 27566   7.4818  52.0734 0011317 322.5609 316.7791  1.00276956 67549'
    tleo = line_to_tle( L1, L2 )
    tdate = astropy.time.Time( tleo.jdsatepoch + 0.6, format='jd' )

    # pick some dates
    tdates = astropy.time.Time( tleo.jdsatepoch + np.arange(0,86400*5,300), format='jd')

    # --------------------------------------------------- LOOP ---------------------------------------------------
    for td in tdates:
        frames = tles_to_frame( [tleo],  td )
        gcrs, truth = frames['GCRS'], frames['truth']
        print("----------------------------------------- {} ----------------------------------------- ".format(td))

        # sensors loop
        for S in as_sensors:
            out = S.makeObs( gcrs, truth, justViewable=True)
            if out is None:  
                print("No obs for {}".format(S.name))
                continue
            print(S)
            print(out)
    # --------------------------------------------------- LOOP ---------------------------------------------------
