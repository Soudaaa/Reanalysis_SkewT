import xarray as xr
import metpy
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
from metpy.units import units
from datetime import datetime
import numpy as np
#import sharppy.plot.skew as spskew
from plots import plot_skewt
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore
from geopy.geocoders import Nominatim
import cdsapi
from urllib.request import urlopen

def retrieve_ERA5(lon = None, lat = None, datetime = None):
    
    if datetime.month < 10:
        month = str(0) + str(datetime.month)
    else:
        month = str(datetime.month)
    if datetime.day < 10:
        day = str(0) + str(datetime.day)
    else:
        day = str(datetime.day)
    if datetime.hour < 10:
        hour = str(0) + str(datetime.hour)
    else:
        hour = str(datetime.hour)        
    
    c = cdsapi.Client()
        
    file_single_levels = c.retrieve('reanalysis-era5-single-levels',
                                    {'product_type': 'reanalysis',
                                     'format': 'netcdf',
                                     'variable': ['10m_u_component_of_wind','10m_v_component_of_wind','2m_dewpoint_temperature',
                                                  '2m_temperature','land_sea_mask','mean_sea_level_pressure',
                                                  'sea_ice_cover','sea_surface_temperature','skin_temperature',
                                                  'snow_depth','soil_temperature_level_1','soil_temperature_level_2',
                                                  'soil_temperature_level_3','soil_temperature_level_4','surface_pressure',
                                                  'volumetric_soil_water_layer_1','volumetric_soil_water_layer_2',
                                                  'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'zero_degree_level'],
                                     'year': f'{datetime.year}',
                                     'month': [month],
                                     'day': [day],
                                     'time': [hour + ':00:00'],
                                     'area': [-10, -80, -50, -35]})

    file_pressure_levels = c.retrieve('reanalysis-era5-pressure-levels',
                                      {'product_type': 'reanalysis',
                                       'format': 'netcdf',
                                       'variable': ['divergence', 'geopotential', 'potential_vorticity',
                                                    'relative_humidity', 'specific_humidity', 'temperature',
                                                    'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                                                    'vorticity'],
                                       'pressure_level': ['10','20', '30', '50', '70', '100', '125',
                                                          '150', '175', '200', '225', '250', '300',
                                                          '350', '400', '450', '500', '550', '600',
                                                          '650', '700', '750', '775', '800', '825',
                                                          '850', '875', '900', '925', '950', '975','1000'],
                                       'year': f'{datetime.year}',
                                       'month': [month],
                                       'day': [day],
                                       'time': [hour + ':00:00'],
                                       'area': [-10, -80, -50,-35]})
    
    f_sfc = urlopen(file_single_levels.location)
    f_pl = urlopen(file_pressure_levels.location)
    ds1 = xr.open_dataset(f_sfc.read()).sel(longitude = lon, latitude = lat, method = 'nearest')
    ds2 = xr.open_dataset(f_pl.read()).sel(longitude = lon, latitude = lat, method = 'nearest')
    ds_concat = xr.merge([ds1, ds2])
    
    lat_grid = ds_concat.coords['latitude'].values.astype('float')
    lon_grid = ds_concat.coords['longitude'].values.astype('float')
    
    p_lev = ds_concat.coords['level'].values * units.hPa
    p_sfc = (ds_concat.variables['sp'].values * units.Pa).to('hPa')

    z = (ds_concat.variables['z'].values / 10) * units.m
    z_sfc = mpcalc.pressure_to_height_std(p_sfc).to('m')
    #hgt_AGL = z - z[0]

    T_lev = (ds_concat.variables['t'].values * units.K).to(units.degC)
    T_sfc = (ds_concat.variables['t2m'].values * units.K).to(units.degC)

    q_p = ds_concat.variables['q'].values
    Td_lev =  mpcalc.dewpoint_from_specific_humidity(p_lev, T_lev, q_p)
    Td_sfc = (ds_concat.variables['d2m'].values * units.K).to(units.degC)
    hgt_0c = np.round(ds_concat.variables['deg0l'].values)
    
    u_lev = (ds_concat.variables['u'].values * units('m/s')).to('kt')
    u_sfc = (ds_concat.variables['u10'].values * units('m/s')).to('kt')

    v_lev = ds_concat.variables['v'].values * units('m/s').to('kt')
    v_sfc = (ds_concat.variables['v10'].values * units('m/s')).to('kt')
    
    p_sounding = np.sort(np.append(p_lev, p_sfc))
    ind = np.where(p_sounding >= p_sfc)[0]
    hgt_sounding = np.insert(z.magnitude, ind[0], z_sfc.magnitude) * units('m')
    T_sounding = np.insert(T_lev.magnitude, ind[0], T_sfc.magnitude)
    Td_sounding = np.insert(Td_lev.magnitude, ind[0], Td_sfc.magnitude) * units('degC')
    u_sounding = (np.insert(u_lev.magnitude, ind[0], u_sfc.magnitude))
    v_sounding = (np.insert(v_lev.magnitude, ind[0], v_sfc.magnitude))

    p_skewt = p_sounding[p_sounding <= p_sfc][::-1]
    hgt_skewt = np.sort(hgt_sounding[p_sounding <= p_sfc])
    hgt_AGL = (hgt_skewt- hgt_skewt[0])
    T_skewt = (T_sounding[p_sounding <= p_sfc] * units.degC)[::-1]
    Td_skewt = (Td_sounding[p_sounding <= p_sfc].to('degC'))[::-1]
    u_skewt = (u_sounding[p_sounding <= p_sfc] * units.kt)[::-1]
    v_skewt = (v_sounding[p_sounding <= p_sfc] * units.kt)[::-1]

    #wspeed_skewt = mpcalc.wind_speed(u_skewt, v_skewt)
    #wdir_skewt = mpcalc.wind_direction(u_skewt, v_skewt)
    
    return lon_grid, lat_grid, p_skewt, hgt_AGL, T_skewt, Td_skewt, u_skewt, v_skewt

def retrieve_CFSRv2(lon = None, lat = None, datetime = None):
    
    if datetime.month < 10:
        month = str(0) + str(datetime.month)
    else:
        month = str(datetime.month)
    if datetime.day < 10:
        day = str(0) + str(datetime.day)
    else:
        day = str(datetime.day)
    if datetime.hour < 10:
        hour = str(0) + str(datetime.hour)
    else:
        hour = str(datetime.hour)
        
    cat = TDSCatalog(f'https://www.ncei.noaa.gov/thredds/catalog/model-cfs_v2_anl_6h_pgb/{datetime.year}/{datetime.year}' + month + f'/{datetime.year}' + month + day + '/'
                      f'catalog.xml?dataset=cfs_v2_anl_6h_pgb/{datetime.year}/{datetime.year}' + month + f'/{datetime.year}' + month + day + '/cdas1.t' + hour + 'z.pgrbh01.grib2')
    
    cat_dataset = cat.datasets[0]
    ncss = cat_dataset.subset()
    query = ncss.query()
    query.lonlat_point(lon, lat)
    query.variables('Geopotential_height_isobaric', 'Geopotential_height_surface', 'Potential_temperature_sigma', 'Specific_humidity_isobaric',
                    'Pressure_msl', 'Pressure_surface', 'Relative_humidity_sigma', 'Temperature_sigma', 'Temperature_isobaric',
                    'v-component_of_wind_sigma', 'u-component_of_wind_sigma', 'v-component_of_wind_isobaric', 'u-component_of_wind_isobaric')
    data = ncss.get_data(query)
    
    lon_grid = data[0]['lon'][0]
    lat_grid = data[0]['lat'][0]
    
    p_lev = (data[2]['vertCoord'][::-1] * units.Pa).to('hPa')
    p_sfc = (data[1]['Pressure_surface'] * units.Pa).to('hPa')
    
    z = data[2]['Geopotential_height_isobaric'][::-1] * units.m
    z_sfc = data[1]['Geopotential_height_surface'] * units.m
    
    T_lev = (data[2]['Temperature_isobaric'][::-1] * units.K).to(units.degC)
    T_sfc = (data[0]['Temperature_sigma'] * units.K).to(units.degC)
    
    q_p = data[2]['Specific_humidity_isobaric'][::-1]
    Td_lev = mpcalc.dewpoint_from_specific_humidity(p_lev, T_lev, q_p)
    Td_sfc = mpcalc.dewpoint_from_relative_humidity(T_sfc, data[0]['Relative_humidity_sigma']/100)
    
    u_lev = (data[2]['u-component_of_wind_isobaric'][::-1] * units('m/s')).to('kt')
    u_sfc = (data[0]['u-component_of_wind_sigma'] * units('m/s')).to('kt')
    
    v_lev = (data[2]['v-component_of_wind_isobaric'][::-1] * units('m/s')).to('kt')
    v_sfc = (data[0]['v-component_of_wind_sigma'] * units('m/s')).to('kt')
    
    p_sounding = np.sort(np.append(p_lev, p_sfc))[::-1].to('hPa')
    ind = np.where(p_sounding >= p_sfc)[0]
    hgt_sounding = np.insert(z.magnitude, ind[0], z_sfc.magnitude) * units('m')
    T_sounding = np.insert(T_lev.magnitude, ind[0], T_sfc.magnitude)
    Td_sounding = np.insert(Td_lev.magnitude, ind[0], Td_sfc.magnitude) * units('degC')
    u_sounding = (np.insert(u_lev.magnitude, ind[0], u_sfc.magnitude))
    v_sounding = (np.insert(v_lev.magnitude, ind[0], v_sfc.magnitude))
    
    p_skewt = p_sounding[p_sounding <= p_sfc]
    hgt_skewt = np.sort(hgt_sounding[p_sounding <= p_sfc])
    hgt_AGL = (hgt_skewt- hgt_skewt[0])
    T_skewt = T_sounding[p_sounding <= p_sfc]
    Td_skewt = Td_sounding[p_sounding <= p_sfc]
    u_skewt = u_sounding[p_sounding <= p_sfc] 
    v_skewt = v_sounding[p_sounding <= p_sfc]

    #wspeed_skewt = mpcalc.wind_speed(u_skewt, v_skewt)
    #wdir_skewt = mpcalc.wind_direction(u_skewt, v_skewt)

    return lon_grid, lat_grid, p_skewt, hgt_AGL, T_skewt, Td_skewt, u_skewt, v_skewt

def retrieve_CFSR(lon = None, lat = None, datetime = None):
    
    if datetime.month < 10:
        month = str(0) + str(datetime.month)
    else:
        month = str(datetime.month)
    if datetime.day < 10:
        day = str(0) + str(datetime.day)
    else:
        day = str(datetime.day)
    if datetime.hour < 10:
        hour = str(0) + str(datetime.hour)
    else:
        hour = str(datetime.hour)
        
    cat = TDSCatalog(f'https://www.ncei.noaa.gov/thredds/catalog/model-cfs_reanl_6h_pgb/{datetime.year}/{datetime.year}' + month + f'/{datetime.year}' + month + day + '/'
                 f'catalog.html?dataset=cfs_reanl_6h_pgb/{datetime.year}/{datetime.year}' + month + f'/{datetime.year}' + month + day + f'/pgbhnl.gdas.{datetime.year}' + month + day + hour + '.grb2') 
    
    cat_dataset = cat.datasets[0]
    ncss = cat_dataset.subset()
    query = ncss.query()
    query.lonlat_point(lon, lat)
    query.variables('Geopotential_height_isobaric', 'Geopotential_height_surface', 'Potential_temperature_sigma', 'Specific_humidity_isobaric',
                    'Pressure_msl', 'Pressure_surface', 'Relative_humidity_sigma', 'Temperature_sigma', 'Temperature_isobaric',
                    'v-component_of_wind_sigma', 'u-component_of_wind_sigma', 'v-component_of_wind_isobaric', 'u-component_of_wind_isobaric')
    data = ncss.get_data(query)
    
    lon_grid = data[0]['lon'][0]
    lat_grid = data[0]['lat'][0]
    
    p_lev = (data[2]['vertCoord'][::-1] * units.Pa).to('hPa')
    p_sfc = (data[1]['Pressure_surface'] * units.Pa).to('hPa')
    
    z = data[2]['Geopotential_height_isobaric'][::-1] * units.m
    z_sfc = data[1]['Geopotential_height_surface'] * units.m
    
    T_lev = (data[2]['Temperature_isobaric'][::-1] * units.K).to(units.degC)
    T_sfc = (data[0]['Temperature_sigma'] * units.K).to(units.degC)
    
    q_p = data[2]['Specific_humidity_isobaric'][::-1]
    Td_lev = mpcalc.dewpoint_from_specific_humidity(p_lev, T_lev, q_p)
    Td_sfc = mpcalc.dewpoint_from_relative_humidity(T_sfc, data[0]['Relative_humidity_sigma']/100)
    
    u_lev = (data[2]['u-component_of_wind_isobaric'][::-1] * units('m/s')).to('kt')
    u_sfc = (data[0]['u-component_of_wind_sigma'] * units('m/s')).to('kt')
    
    v_lev = (data[2]['v-component_of_wind_isobaric'][::-1] * units('m/s')).to('kt')
    v_sfc = (data[0]['v-component_of_wind_sigma'] * units('m/s')).to('kt')
    
    p_sounding = np.sort(np.append(p_lev, p_sfc))[::-1].to('hPa')
    ind = np.where(p_sounding >= p_sfc)[0]
    hgt_sounding = np.insert(z.magnitude, ind[0], z_sfc.magnitude) * units('m')
    T_sounding = np.insert(T_lev.magnitude, ind[0], T_sfc.magnitude)
    Td_sounding = np.insert(Td_lev.magnitude, ind[0], Td_sfc.magnitude) * units('degC')
    u_sounding = (np.insert(u_lev.magnitude, ind[0], u_sfc.magnitude))
    v_sounding = (np.insert(v_lev.magnitude, ind[0], v_sfc.magnitude))
    
    p_skewt = p_sounding[p_sounding <= p_sfc]
    hgt_skewt = np.sort(hgt_sounding[p_sounding <= p_sfc])
    hgt_AGL = (hgt_skewt- hgt_skewt[0])
    T_skewt = T_sounding[p_sounding <= p_sfc]
    Td_skewt = Td_sounding[p_sounding <= p_sfc]
    u_skewt = u_sounding[p_sounding <= p_sfc] 
    v_skewt = v_sounding[p_sounding <= p_sfc]

    #wspeed_skewt = mpcalc.wind_speed(u_skewt, v_skewt)
    #wdir_skewt = mpcalc.wind_direction(u_skewt, v_skewt)

    return lon_grid, lat_grid, p_skewt, hgt_AGL, T_skewt, Td_skewt, u_skewt, v_skewt

def sel_data(reanalysis = 'ERA5', cidade = None, estado = None, dt = None):
    geoloc = Nominatim(user_agent="Your Name")
    location = geoloc.geocode(cidade + ',' + estado)
    time = datetime.strptime(dt, '%Y%m%d%H')
    if reanalysis=='ERA5':
        lon_grid, lat_grid, p_skew, hgt_agl, T_skew, Td_skew, u_skew, v_skew = retrieve_ERA5(location.longitude, location.latitude, time)
    elif reanalysis=='CFSR':
        lon_grid, lat_grid, p_skew, hgt_agl, T_skew, Td_skew, u_skew, v_skew = retrieve_CFSR(location.longitude, location.latitude, time)
    elif reanalysis=='CFSRv2':
        lon_grid, lat_grid, p_skew, hgt_agl, T_skew, Td_skew, u_skew, v_skew = retrieve_CFSRv2(location.longitude, location.latitude, time)
        
    return lon_grid, lat_grid, p_skew, hgt_agl, T_skew, Td_skew, u_skew, v_skew, time, reanalysis