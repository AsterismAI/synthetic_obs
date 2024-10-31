import numpy as np
import astropy.coordinates
import astropy.units as u
import astropy.time

# (KNW) : these are the old versions that were based on AstroPy frames 
#       : not sure which we'll use so keep these
## ----------------------------------------------------------------------------------------------------- 
#def solar_angle( gcrs_frame, sensor_gcrs ):
#    ''' 
#    from the sensor, view angle between the object and the sun (solar exclusion calculation)
#    '''
#    sungcrs = astropy.coordinates.get_sun( gcrs_frame.obstime )
#    # in matrix
#    sat_pos  = gcrs_frame.cartesian.xyz.to_value(u.km).T
#    ones     = np.ones( (sat_pos.shape[0],3) )
#    sun_pos  = sungcrs.cartesian.xyz.to_value(u.km).T  * ones
#    obs_pos  = sensor_gcrs.cartesian.xyz.to_value(u.km).T  * ones
#    sat_look = sat_pos - obs_pos
#    sun_look = sun_pos - obs_pos
#    sat_look = sat_look / np.linalg.norm( sat_look, axis=1 )[:,np.newaxis]
#    sun_look = sun_look / np.linalg.norm( sun_look, axis=1 )[:,np.newaxis]
#    return np.degrees( np.arccos( np.einsum('ij,ij->i',sat_look,sun_look) ) )
#
## ----------------------------------------------------------------------------------------------------- 
#def solar_phase( gcrs_frame, sensor_gcrs ):
#    '''
#    this is slightly more complex, but we'll stick to the vector math in GCRS to get this
#    generate two look vectors from the object in question (1) to sensor (2) to sun
#    and get the angle between the two
#    '''
#    N = gcrs_frame.shape[0]
#    sensor_gcrs = sensor_gcrs.cartesian.xyz.to_value(u.km).T * np.ones((N,3))
#    sunloc     = astropy.coordinates.get_sun( gcrs_frame.obstime ).gcrs.cartesian.xyz.to_value(u.km).T * np.ones((N,3))
#    sat_to_sun = sunloc - gcrs_frame.cartesian.xyz.to_value(u.km).T
#    sat_to_sen = sensor_gcrs - gcrs_frame.cartesian.xyz.to_value(u.km).T
#    sat_to_sun = sat_to_sun / np.linalg.norm( sat_to_sun, axis=1 )[:,np.newaxis]
#    sat_to_sen = sat_to_sen / np.linalg.norm( sat_to_sen, axis=1 )[:,np.newaxis]
#    return np.degrees( np.arccos( np.einsum('ij,ij->i',sat_to_sun,sat_to_sen) ) )
#
#
## ----------------------------------------------------------------------------------------------------- 
#def solar_angle( gcrs_frame, sensor_gcrs, sun_gcrs ):
#    ''' 
#    KNW: these are all Nx3 matrices generated outside this function
#    where N = number of objects (or number of time-steps )
#    it is agnostic
#    '''
#    # get look vectors
#    sat_look = gcrs_frame - sensor_gcrs 
#    sun_look = sun_gcrs - sensor_gcrs 
#    # normalize (KNW: we could probably save one normalization here.. keep for now)
#    sat_look = sat_look / np.linalg.norm( sat_look )
#    sun_look = sun_look / np.linalg.norm( sun_look )
#    sun_look = sun_look * np.ones( sat_look.shape )
#    # return angle 
#    return np.degrees( np.arccos( np.einsum('ij,ij->i',sat_look,sun_look) ) )
#
## ----------------------------------------------------------------------------------------------------- 
#def solar_phase( gcrs_frame, sensor_gcrs, sun_gcrs ):
#    '''
#    this is slightly more complex, but we'll stick to the vector math in GCRS to get this
#    generate two look vectors from the object in question (1) to sensor (2) to sun
#    and get the angle between the two
#    '''
#    # get look vectors
#    sat_to_sun = sun_gcrs - gcrs_frame
#    sat_to_sen = sensor_gcrs - gcrs_frame 
#    # normalize 
#    sat_to_sun = sat_to_sun / np.linalg.norm( sat_to_sun, axis=1 )[:,np.newaxis]
#    sat_to_sen = sat_to_sen / np.linalg.norm( sat_to_sen, axis=1 )[:,np.newaxis]
#    # return angle 
#    return np.degrees( np.arccos( np.einsum('ij,ij->i',sat_to_sun,sat_to_sen) ) )
#
# ----------------------------------------------------------------------------------------------------- 
# (KNW) cleaned up version July 20, 2021
def solar_calcs( gcrs_frame, sensor_gcrs, sun = None ):
    '''
    astropy.coordinates.GCRS :
    gcrs_frame  : objects in the simulation
    sensor_gcrs : where the sensor currently is

    Returns : (solar_angle, solar_phase)
    we can re-use sun position for both solar_phase and solar_angle, so wrap them here
    '''
    
    if sun is None: sun_gcrs = astropy.coordinates.get_sun( gcrs_frame.obstime )
    else: sun_gcrs = sun
    sun_gcrs_np    = sun_gcrs.cartesian.xyz.to_value(u.km).T
    sensor_gcrs_np = sensor_gcrs.cartesian.xyz.to_value(u.km).T
    obj_gcrs_np   = gcrs_frame.cartesian.xyz.to_value(u.km).T 

    # solar phase// angle between (obj)-->(sen) and (obj)-->(sun)
    sun_to_obj     = sun_gcrs_np    - obj_gcrs_np
    sun_to_obj     = sun_to_obj / np.linalg.norm(sun_to_obj, axis=1 )[:,np.newaxis]
    obj_to_sens    = sensor_gcrs_np - obj_gcrs_np
    obj_to_sens    = obj_to_sens / np.linalg.norm(obj_to_sens, axis=1)[:,np.newaxis]
    solar_phase = np.degrees( np.arccos( np.einsum('ij,ij->i',sun_to_obj, obj_to_sens) ) )

    # solar angle angle between (sen)-->(sun) and (sen)-->(obj)
    sen_to_sun     = sun_gcrs_np - sensor_gcrs_np
    sen_to_sun     = sen_to_sun * np.ones( shape=obj_gcrs_np.shape ) # scale to objects
    sen_to_sun     = sen_to_sun / np.linalg.norm( sen_to_sun, axis=1 )[:,np.newaxis]
    sen_to_obj     = - obj_to_sens # (from above)
    solar_angle = np.degrees( np.arccos( np.einsum('ij,ij->i',sen_to_sun, sen_to_obj ) ) )

    return solar_angle, solar_phase, sun_gcrs
    # call functions with numpy arrays
    #ang   = solar_angle( obj_gcrs_np, sensor_gcrs_np, sun_gcrs_np )
    #phase = solar_phase( obj_gcrs_np, sensor_gcrs_np, sun_gcrs_np )
    #return ang,phase,sun_gcrs

# =====================================================================================================
if __name__ == "__main__":
    import os
    import sys
    import json
    libpath = os.path.abspath('../libs')
    if libpath in sys.path: pass
    else: sys.path.append( libpath )
    from tle_to_frame import tle_to_frame
    import sensor_model 

    # ------------------  LOAD IN SENSORS -----------------------
    with open('./data/geo_sensors_subset_reformatted.json') as F: as_sensors = json.load(F).values()
    as_sensors = [sensor_model.sensor_astro_standards(S,S['location']['DESC']) for S in as_sensors]
    print('Loaded {:04d} sensors'.format( len(as_sensors) ) )
    # ------------------  space based are different! -------------------
    # here, we'll pass in the state based on the universal state and gin up obs accordingly
    ors = sensor_model.sensor_ors5(name='ORS5')
    # ------------------  append ORS to the list -----------------------
    as_sensors.append( ors )
    # ------------------------------- LOAD TLE AND TEST -------------------------------
    L1 = '1 27566U 02055A   21130.76031345  .00000095  00000-0  00000-0 0  9998'
    L2 = '2 27566   7.4818  52.0734 0011317 322.5609 316.7791  1.00276956 67549'
    frames = tle_to_frame( L1, L2 )
    gcrs, pdf = frames['GCRS'], frames['pandas']

    #self.getSensorLocGCRS( 
    S = as_sensors[0] 
    sen_gcrs = S.getSensorLocGCRS( gcrs.obstime )
    solar_angle, solar_phase = solar_calcs( gcrs, sen_gcrs ) 
