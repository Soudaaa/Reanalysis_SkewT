import xarray as xr
import metpy
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
from metpy.units import units
from datetime import datetime
import numpy as np
import pandas as pd
from plots import plot_skewt
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore
from geopy.geocoders import Nominatim
import cdsapi
from urllib.request import urlopen

def retrieve_ERA5_full(datetime = None, skewt = False):
    
    model_table = pd.read_csv('table.csv')
    
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
    
    model_data_sfc = c.retrieve('reanalysis-era5-single-levels',
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
                                 'month': month,
                                 'day': day,
                                 'time': ['00:00', '01:00', '02:00',
                                          '03:00', '04:00', '05:00',
                                          '06:00', '07:00', '08:00',
                                          '09:00', '10:00', '11:00',
                                          '12:00', '13:00', '14:00',
                                          '15:00', '16:00', '17:00',
                                          '18:00', '19:00', '20:00',
                                          '21:00', '22:00', '23:00',],
                                 'area': [-10, -80, -50, -35]})
        
    open_data_sfc = urlopen(model_data_sfc.location)
    model_sfc = xr.open_dataset(open_data_sfc.read())
        
    model_data_lev = c.retrieve('reanalysis-era5-pressure-levels',
                                {'product_type': 'reanalysis',
                                 'format': 'netcdf',
                                 'variable': ['divergence', 'geopotential', 'potential_vorticity',
                                              'relative_humidity', 'specific_humidity', 'temperature',
                                              'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                                              'vorticity'],
                                 'pressure_level': ['10','20', '30', '50',
                                                    '70', '100', '125','150',
                                                    '175', '200','225', '250',
                                                    '300','350', '400', '450',
                                                    '500', '550', '600', '650',
                                                    '700', '750', '775', '800',
                                                    '825', '850', '875', '900',
                                                    '925', '950', '975', '1000',],
                                 'year':  f'{datetime.year}',
                                 'month': month,
                                 'day': day,
                                 'time': ['00:00', '01:00', '02:00',
                                          '03:00', '04:00', '05:00',
                                          '06:00', '07:00', '08:00',
                                          '09:00', '10:00', '11:00',
                                          '12:00', '13:00', '14:00', 
                                          '15:00', '16:00', '17:00',
                                          '18:00', '19:00', '20:00',
                                          '21:00', '22:00', '23:00'],
                                 'area': [-10, -80, -50, -35]})
        
    open_data_lev = urlopen(model_data_lev.location)
    model_lev = xr.open_dataset(open_data_lev.read())
    model_data = xr.merge([model_sfc, model_lev])
    
    return model_data

def retrieve_ERA5(lon = None, lat = None, datetime = None, skewt = False):
    
    model_table = pd.read_csv('table.csv')
    
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

    model_data = c.retrieve('reanalysis-era5-complete',
                            {'date'    : f'{datetime.year}' + '-' + month + '-' + day,            # The hyphens can be omitted
                             'levelist': '1/to/137',          # 1 is top level, 137 the lowest model level in ERA5. Use '/' to separate values.
                             'levtype' : 'ml',
                             'param'   : '75/76/77/129/130/131/132/133/135/138/152/155/203/246/247/248',                   # Full information at https://apps.ecmwf.int/codes/grib/param-db/
                             'stream'  : 'oper',                  # Denotes ERA5. Ensemble members are selected by 'enda'
                             'time'    : hour + ':00:00',         # You can drop :00:00 and use MARS short-hand notation, instead of '00/06/12/18'
                             'type'    : 'an',
                             'area'    : [lon + 1., lon - 1., lat - 1., lat + 1],          # North, West, South, East. Default: global
                             'grid'    : '0.1/0.1',
                             'format'  : 'netcdf'# Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
                            })     # Output file. Adapt as you wish.

    
    open_data = urlopen(model_data.location)

    model_data = xr.open_dataset(open_data.read())
    model_data = model_data.sel(time = model_data['time'].values[0], longitude = lon, latitude = lat, method = 'nearest')
    
    lat_grid = model_data.coords['latitude'].values.astype('float')
    lon_grid = model_data.coords['longitude'].values.astype('float')
    
    sp = np.exp(model_data.variables['lnsp'].values[0])
    
    ph = []
    for a,b in zip(model_table['a [Pa]'].values, model_table['b'].values):
        ph.append(a + b*sp)
        
    ph = np.asarray(ph)
    
    pf = np.zeros_like(ph)
    for i in range(len(ph)):
        if i == 0:
            pf[i] = ph[i]
        else:
            pf[i] = (ph[i] + ph[i-1])/2
    
    model_data['pml'] = np.delete(pf.astype(np.float32), 0)
    model_data['pml'].attrs = {'units' : 'Pa','long_name':'pressure','standard_name':'air_pressure','positive':'down'}
    model_data['hgt_msl'] = mpcalc.pressure_to_height_std(model_data['pml'].values * units('Pa'))
    model_data['hgt_msl'].attrs = {'units' : 'km','long_name': 'geopotential height', 'standard_name':'geopotential_height','positive':'up'}
    model_data['td'] = mpcalc.dewpoint_from_specific_humidity((model_data['pml'].values * units.Pa).to('hPa'), (model_data['t'].values * units.K).to('degC'), model_data['q'].values)
    model_data['td'].attrs = {'units' : 'degree Celsius', 'long_name': 'dewpoint', 'standard_name':'dewpoint','positive':'down'}

    p_skewt = ((model_data['pml'][::-1].values * units.Pa).to('hPa')).magnitude
    hgt_skewt = ((model_data['hgt_msl'][::-1].values * units.km).to('m')).magnitude
    T_skewt = ((model_data['t'][::-1].values * units.K).to('degC')).magnitude
    Td_skewt = model_data['td'][::-1].values
    u_skewt = (model_data['u'][::-1].values*units('m/s').to('kt')).magnitude
    v_skewt = (model_data['v'][::-1].values*units('m/s').to('kt')).magnitude
    
    return lon_grid, lat_grid, p_skewt, hgt_skewt, T_skewt, Td_skewt, u_skewt, v_skewt

def retrieve_CFSRv2_full(datetime = None):
    
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
    query.lonlat_box(-80, -35, -50, -10)
    query.variables('Geopotential_height_isobaric', 'Geopotential_height_surface', 'Relative_humidity_isobaric', 'Relative_humidity_height_above_ground',
                    'Pressure_msl', 'Pressure_surface',  'Temperature_isobaric', 'Temperature_height_above_ground', 'Dewpoint_temperature_height_above_ground',
                    'v-component_of_wind_height_above_ground', 'u-component_of_wind_height_above_ground', 'v-component_of_wind_isobaric', 'u-component_of_wind_isobaric')
    data = ncss.get_data(query)
    
    return data

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
    query.lonlat_box(lon -1, lon + 1., lat - 1., lat + 1)
    query.variables('Geopotential_height_isobaric', 'Geopotential_height_surface', 'Relative_humidity_isobaric', 'Relative_humidity_height_above_ground',
                    'Pressure_msl', 'Pressure_surface',  'Temperature_isobaric', 'Temperature_height_above_ground', 'Dewpoint_temperature_height_above_ground',
                    'v-component_of_wind_height_above_ground', 'u-component_of_wind_height_above_ground', 'v-component_of_wind_isobaric', 'u-component_of_wind_isobaric')
    dataset = ncss.get_data(query)

    data = xr.open_dataset(NetCDF4DataStore(dataset)).metpy.parse_cf()
    try:
        ds = data.sel(time = data['time'].values[0], lon = lon, lat = lat, height_above_ground1 = '10', height_above_ground = '2', method = 'nearest')
    except:
        ds = data.sel(time1 = data['time1'].values[0], lon = lon, lat = lat, height_above_ground1 = '10', height_above_ground = '2', method = 'nearest')
    lon_grid = ds.lon.values
    lat_grid = ds.lat.values

    p_lev = (ds.coords['isobaric3'][::-1].values / 100 * units.hPa)
    p_sfc = (ds.variables['Pressure_msl'].values * units.Pa).to('hPa')
    z = ds.variables['Geopotential_height_isobaric'].values 
    z_sfc = ds.variables['Geopotential_height_surface'].values
    T_lev = (ds.variables['Temperature_isobaric'][::-1].values * units.K).to('degC')
    T_sfc = (ds.variables['Temperature_height_above_ground'].values * units.K).to('degC')
    RH = ds.variables['Relative_humidity_isobaric'][::-1].values
    Td_lev = mpcalc.dewpoint_from_relative_humidity(T_lev.to('K'), RH/100)
    Td_sfc = (ds.variables['Dewpoint_temperature_height_above_ground'].values * units.K).to('degC')
    u_lev = (ds.variables['u-component_of_wind_isobaric'][::-1].values * units('m/s')).to('kt')
    v_lev = (ds.variables['v-component_of_wind_isobaric'][::-1].values * units('m/s')).to('kt')
    u_sfc = (ds.variables['u-component_of_wind_height_above_ground'].values * units('m/s')).to('kt')
    v_sfc = (ds.variables['v-component_of_wind_height_above_ground'].values * units('m/s')).to('kt')

    p_sounding = np.sort(np.append(p_lev, p_sfc))[::-1].to('hPa')
    ind = np.where(p_sounding >= p_sfc)[0]
    hgt_sounding = np.insert(z, ind[0], z_sfc) * units('m')
    T_sounding = np.insert(T_lev, ind[0], T_sfc)
    Td_sounding = np.insert(Td_lev, ind[0], Td_sfc) 
    u_sounding = (np.insert(u_lev, ind[0], u_sfc))
    v_sounding = (np.insert(v_lev, ind[0], v_sfc))
    
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
    query.variables('Geopotential_height_isobaric', 'Geopotential_height_surface', 'Relative_humidity_isobaric', 'Relative_humidity_height_above_ground',
                    'Pressure_msl', 'Pressure_surface',  'Temperature_isobaric', 'Temperature_height_above_ground',
                    'v-component_of_wind_height_above_ground', 'u-component_of_height_above_ground', 'v-component_of_wind_isobaric', 'u-component_of_wind_isobaric')
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
    """
    
    Parameters
    ----------
    reanalysis : str
        Renalysis model to extract pseudosounding (ERA5, CFSR, CFSRv2). Default is ERA5.
    cidade : str
        City name to which the closest grid point will be used to generate de vertical profile.
    estado : str
        Estate of the city where profile will be extracted from
    dt : str
        date and time of interest in the "YYYYMDH" format
    
    """
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