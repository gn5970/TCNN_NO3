import numpy as np
import cartopy
import numpy.polynomial.polynomial as poly
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import zscore

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import zscore

# Define file path and load data
infile = '/Users/navarra/Desktop/wod_1955-2023_anom.nc'
data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data['time'] = pd.date_range("1955-01-01", periods=828, freq="M")
#data=data.isel(time=slice(120, 768))
data=data.isel(time=slice(288, 768))
# Define geographical bounds
min_lon, max_lon = -180, 180
min_lat, max_lat = -90, -50
min_depth, max_depth = 0, 100

# Apply mask
mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data = data.where(mask_lon & mask_lat & mask_depth, drop=True)
data=data.mean('depth')
# Create a mask for non-null values
mask = data.notnull()

# Define helper functions
def remove_time_mean(x):
    return x - x.mean(dim='time', skipna=True)

def detrend_dim(da, dim, deg=1):
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def wgt_areaave(indat, latS, latN, lonW, lonE):
    lat = indat.lat
    lon = indat.lon

    if (((lonW < 0) or (lonE < 0)) and (lon.values.min() > -1)):
        anm = indat.assign_coords(lon=((lon + 180) % 360 - 180))
        lon = ((lon + 180) % 360 - 180)
    else:
        anm = indat

    iplat = lat.where((lat >= latS) & (lat <= latN), drop=True)
    iplon = lon.where((lon >= lonW) & (lon <= lonE), drop=True)
    wgt = np.cos(np.deg2rad(lat))
    odat = anm.sel(lat=iplat, lon=iplon).weighted(wgt).mean(("lon", "lat"), skipna=True)
    return odat

anomaly_no3 = wgt_areaave(data.no3, -90, -50, -180, 180)
anomaly_no3 = anomaly_no3 / anomaly_no3.std()
no3_glb_avg=np.array(anomaly_no3)
# Calculate z-scores and exclude outliers
mean = np.nanmean(no3_glb_avg)
standard_deviation = np.nanstd(no3_glb_avg)
distance_from_mean = abs(no3_glb_avg - mean)
max_deviations = 3
not_outlier1 = distance_from_mean < max_deviations * standard_deviation
outlier=distance_from_mean > max_deviations * standard_deviation
no3_glb_avg=no3_glb_avg[not_outlier1]
time=data.time.values[not_outlier1]

no3_glb_avg=xr.DataArray(no3_glb_avg)#detrend_dim(xr.DataArray(no3_glb_avg),dim='dim_0')
no3_glb_avg = no3_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
no3_glb_avg=detrend_dim(xr.DataArray(no3_glb_avg),dim='time')


# Plot the anomalies with running mean
plt.figure(figsize=(12, 6))
anomaly_no3.isel(time=slice(0,626)).coarsen(time=2).mean().plot()
plt.xlabel("Time", fontsize=20)
plt.ylabel("NO3 anomalies", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("NO3 Anomalies (Running Mean, Window=3) without pre-process", fontsize=22)
plt.tight_layout()
plt.legend()
plt.show()

# Plot the anomalies with running mean
plt.figure(figsize=(12, 6))
no3_glb_avg.isel(time=slice(0,316)).coarsen(time=2).mean().plot()
plt.xlabel("Time", fontsize=20)
plt.ylabel("NO3 anomalies", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("NO3 Anomalies (Running Mean, Window=3)", fontsize=22)
plt.tight_layout()
plt.legend()
plt.show()

# Preprocess data
#temp_weighted_clim = data.temp.isel(time=slice(0,780)).mean("depth").where(mask.temp.isel(time=slice(0,780)).mean("depth")).groupby('time.month').apply(remove_time_mean)
#S_temp = data.temp.mean("depth").isel(time=slice(0,780)).where(mask.temp.mean("depth").isel(time=slice(0,780))).groupby('time.month').apply(remove_time_mean)

anomaly_tos = wgt_areaave(data.temp, -90, -50, -180, 180)
anomaly_tos = anomaly_tos / anomaly_tos.std()
tos_glb_avg=np.array(anomaly_tos)
# Calculate z-scores and exclude outliers
mean = np.nanmean(tos_glb_avg)
standard_deviation = np.nanstd(tos_glb_avg)
distance_from_mean = abs(tos_glb_avg - mean)
max_deviations = 3
not_outlier = distance_from_mean < max_deviations * standard_deviation
outlier=distance_from_mean > max_deviations * standard_deviation
tos_glb_avg=tos_glb_avg[not_outlier1]

time=data.time.values[not_outlier1]

tos_glb_avg=xr.DataArray(tos_glb_avg)#detrend_dim(xr.DataArray(tos_glb_avg),dim='dim_0')
tos_glb_avg = tos_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
tos_glb_avg=detrend_dim(xr.DataArray(tos_glb_avg),dim='time')

# Plot the anomalies without outliers
plt.figure(figsize=(12, 6))
tos_glb_avg.isel(time=slice(0,264)).coarsen(time=2).mean().plot()
plt.xlabel("Time", fontsize=20)
plt.ylabel("SST anomalies", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("SST Anomalies (Deseasonalized, Detrended, and Excluding Outliers)", fontsize=22)
plt.tight_layout()
plt.show()



#salt_weighted_clim = data.salt.isel(time=slice(0,780)).mean("depth").where(mask.salt.isel(time=slice(0,780)).mean("depth")).groupby('time.month').apply(remove_time_mean)
#S_salt = data.salt.mean("depth").isel(time=slice(0,780)).where(mask.salt.mean("depth").isel(time=slice(0,780))).groupby('time.month').apply(remove_time_mean)

#anomaly_salt = wgt_areaave(S_salt, -90, -50, -180, 180)
#anomaly_salt = anomaly_salt / anomaly_salt.std()
#salt_glb_avg=np.array(anomaly_salt)
# Calculate z-scores and exclude outliers
#mean = np.nanmean(salt_glb_avg)
#standard_deviation = np.nanstd(salt_glb_avg)
#distance_from_mean = abs(salt_glb_avg - mean)
#max_deviations = 2
#not_outlier = distance_from_mean < max_deviations * standard_deviation
#outlier=distance_from_mean > max_deviations * standard_deviation
#salt_glb_avg=salt_glb_avg[not_outlier]
#salt_glb_avg=detrend_dim(xr.DataArray(salt_glb_avg),dim='dim_0')


anomaly_salt = wgt_areaave(data.salt, -90, -50, -180, 180)
anomaly_salt = anomaly_salt / anomaly_salt.std()
salt_glb_avg=np.array(anomaly_salt)
# Calculate z-scores and exclude outliers
mean = np.nanmean(salt_glb_avg)
standard_deviation = np.nanstd(salt_glb_avg)
distance_from_mean = abs(salt_glb_avg - mean)
max_deviations = 3
not_outlier = distance_from_mean < max_deviations * standard_deviation
outlier=distance_from_mean > max_deviations * standard_deviation
salt_glb_avg=salt_glb_avg[not_outlier1]
salt_glb_avg=xr.DataArray(salt_glb_avg)#detrend_dim(xr.DataArray(salt_glb_avg),dim='dim_0')
time=data.time.values[not_outlier1]
salt_glb_avg = salt_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
salt_glb_avg=detrend_dim(xr.DataArray(salt_glb_avg),dim='time')

# Plot the anomalies without outliers
plt.figure(figsize=(12, 6))
salt_glb_avg.isel(time=slice(0,316)).coarsen(time=2).mean().plot()
plt.xlabel("Time", fontsize=20)
plt.ylabel("Salinity anomalies", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Salinity Anomalies (Deseasonalized, Detrended, and Excluding Outliers)", fontsize=22)
plt.tight_layout()
plt.show()



A_salt=np.corrcoef(no3_glb_avg.isel(time=slice(0,316)).coarsen(time=2).mean(),salt_glb_avg.isel(time=slice(0,316)).coarsen(time=2).mean())
A_temp=np.corrcoef(no3_glb_avg.isel(time=slice(0,316)).coarsen(time=2).mean(),tos_glb_avg.isel(time=slice(0,316)).coarsen(time=2).mean())
#no3_weighted_clim = data.no3.isel(time=slice(0,780)).mean("depth").where(mask.no3.isel(time=slice(0,780)).mean("depth")).groupby('time.month').apply(remove_time_mean)
#S_no3 = data.no3.mean("depth").isel(time=slice(0,780)).where(mask.no3.mean("depth").isel(time=slice(0,780))).groupby('time.month').apply(remove_time_mean)
#anomaly_no3 = wgt_areaave(S_no3, -90, -50, -180, 180)
#anomaly_no3 = anomaly_no3 / anomaly_no3.std()
#no3_glb_avg=np.array(anomaly_no3)
# Calculate z-scores and exclude outliers
#mean = np.nanmean(no3_glb_avg)
#standard_deviation = np.nanstd(no3_glb_avg)
#distance_from_mean = abs(no3_glb_avg - mean)
#max_deviations = 2
#not_outlier = distance_from_mean < max_deviations * standard_deviation
#outlier=distance_from_mean > max_deviations * standard_deviation
#no3_glb_avg=no3_glb_avg[not_outlier]
#no3_glb_avg=detrend_dim(xr.DataArray(no3_glb_avg),dim='dim_0')


def pdens(S,theta):

    # --- Define constants (Table 1 Column 4, Wright 1997, J. Ocean Tech.)---
    a0 = 7.057924e-4
    a1 = 3.480336e-7
    a2 = -1.112733e-7

    b0 = 5.790749e8
    b1 = 3.516535e6
    b2 = -4.002714e4
    b3 = 2.084372e2
    b4 = 5.944068e5
    b5 = -9.643486e3

    c0 = 1.704853e5
    c1 = 7.904722e2
    c2 = -7.984422
    c3 = 5.140652e-2
    c4 = -2.302158e2
    c5 = -3.079464

    # To compute potential density keep pressure p = 100 kpa
    # S in standard salinity units psu, theta in DegC, p in pascals

    p = 100000.
    alpha0 = a0 + a1*theta + a2*S
    p0 = b0 + b1*theta + b2*theta**2 + b3*theta**3 + b4*S + b5*theta*S
    lambd = c0 + c1*theta + c2*theta**2 + c3*theta**3 + c4*S + c5*theta*S

    pot_dens = (p + p0)/(lambd + alpha0*(p + p0))

    return pot_dens

import numpy as np
import pandas as pd
from scipy.stats import norm

def lagged_correlation(series1, series2, max_lag):
    """
    Calculate the lagged correlation between two time series.
    
    Args:
    - series1: The first time series (numpy array or pandas Series).
    - series2: The second time series (numpy array or pandas Series).
    - max_lag: The maximum lag to consider (integer).
    
    Returns:
    - lags: An array of lag values.
    - correlations: An array of correlation coefficients corresponding to each lag.
    - p_values: An array of p-values corresponding to each correlation coefficient.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []
    p_values = []

    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(series1[:lag], series2[-lag:])[0, 1]
            n = len(series1[:lag])
        elif lag > 0:
            corr = np.corrcoef(series1[lag:], series2[:-lag])[0, 1]
            n = len(series1[lag:])
        else:
            corr = np.corrcoef(series1, series2)[0, 1]
            n = len(series1)
        correlations.append(corr)
        
        # Fisher Z-transformation
        fisher_z = np.arctanh(corr)
        standard_error = 1 / np.sqrt(n - 3)
        z = fisher_z / standard_error
        p_value = 2*(1 - norm.cdf(abs(z)))  # Two-tailed test
        p_values.append(p_value)

    return lags, correlations, p_values

# Assuming no3_glb_avg and salt_glb_avg are your time series
# Ensure that they have the same length
if len(no3_glb_avg) != len(salt_glb_avg):
    raise ValueError("The time series must have the same length")

max_lag = 48  # Define the maximum lag you want to consider
lags, correlations, p_values = lagged_correlation(no3_glb_avg.isel(time=slice(0,264)).coarsen(time=2).mean(), salt_glb_avg.isel(time=slice(0,264)).coarsen(time=2).mean(), max_lag)

# Convert to pandas DataFrame for better visualization
correlation_df = pd.DataFrame({'Lag': lags, 'Correlation': correlations, 'p-value': p_values})

# Plot the lagged correlation
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(correlation_df['Lag'], correlation_df['Correlation'], marker='o', label='Correlation')
plt.axhline(0, color='grey', linestyle='--')
plt.axhline(y=0.196, color='red', linestyle='--', label='Significance Level (p < 0.05)')
plt.axhline(y=-0.196, color='red', linestyle='--')

plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title('Lagged Correlation between NO3 and Salt')
plt.grid(True)

# Highlight significant correlations
significance_level = 0.05
significant = correlation_df['p-value'] < significance_level
plt.scatter(correlation_df['Lag'][significant], correlation_df['Correlation'][significant], color='red', label='Significant (p < 0.05)')

plt.legend()
plt.show()

# Repeat for NO3 and Temperature
lags_temp, correlations_temp, p_values_temp = lagged_correlation(no3_glb_avg.isel(time=slice(0,264)).coarsen(time=2).mean(), tos_glb_avg.isel(time=slice(0,264)).coarsen(time=2).mean(), max_lag)

# Convert to pandas DataFrame for better visualization
correlation_temp_df = pd.DataFrame({'Lag': lags_temp, 'Correlation': correlations_temp, 'p-value': p_values_temp})

# Plot the lagged correlation
plt.figure(figsize=(10, 6))
plt.plot(correlation_temp_df['Lag'], correlation_temp_df['Correlation'], marker='o', label='Correlation')
plt.axhline(0, color='grey', linestyle='--')
plt.axhline(y=0.196, color='red', linestyle='--', label='Significance Level (p < 0.05)')
plt.axhline(y=-0.196, color='red', linestyle='--')

plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title('Lagged Correlation between NO3 and Temperature')
plt.grid(True)

# Highlight significant correlations
significant_temp = correlation_temp_df['p-value'] < significance_level
plt.scatter(correlation_temp_df['Lag'][significant_temp], correlation_temp_df['Correlation'][significant_temp], color='red', label='Significant (p < 0.05)')

plt.legend()
plt.show()


infile='http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/ORAS5/1x1_grid/sossheig/opa0'
data1 = xr.open_dataset(infile)
data1['time'] = pd.date_range("1979-01-15", periods=480, freq="M")

# Define geographical bounds
min_lon, max_lon = 0, 360
min_lat, max_lat = -90, -50
min_depth, max_depth = 0, 50

mask_lon = (data1.lon >= min_lon) & (data1.lon <= max_lon)
mask_lat = (data1.lat >= min_lat) & (data1.lat <= max_lat)
#mask_depth = (data1.lev >= min_depth) & (data1.lev <= max_depth)
data1 = data1.where(mask_lon & mask_lat , drop=True)
#data1=data1.mean('lev')

ssh_clim = data1.sossheig.groupby('time.month').mean(dim='time',skipna=True)
ssh_anom = data1.sossheig.groupby('time.month') - ssh_clim

anomaly_ssh = wgt_areaave(ssh_anom, -90, -50, 0, 360)
anomaly_ssh = anomaly_ssh / anomaly_ssh.std()
ssh_glb_avg=np.array(anomaly_ssh)
ssh_glb_avg=ssh_glb_avg[not_outlier1]
ssh_glb_avg=xr.DataArray(ssh_glb_avg)#detrend_dim(xr.DataArray(buoyancy_glb_avg),dim='dim_0')
ssh_glb_avg = ssh_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
ssh_glb_avg=detrend_dim(xr.DataArray(ssh_glb_avg),dim='time')

#not_outlier = distance_from_mean < max_deviations * standard_deviation
#outlier=distance_from_mean > max_deviations * standard_deviation
#tos_glb_avg=tos_glb_avg[not_outlier1]
infile='http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/ORAS5/1x1_grid/somxl010/opa0'
data1 = xr.open_dataset(infile)
data1['time'] = pd.date_range("1979-01-15", periods=480, freq="M")

mld_clim = data1.somxl010.groupby('time.month').mean(dim='time',skipna=True)
mld_anom = data1.somxl010.groupby('time.month') - mld_clim

anomaly_mld = wgt_areaave(mld_anom, -90, -50, 0, 360)
mld_glb_avg = anomaly_mld / anomaly_mld.std()
mld_glb_avg=np.array(mld_glb_avg)
mld_glb_avg=mld_glb_avg[not_outlier1]
mld_glb_avg=xr.DataArray(mld_glb_avg)#detrend_dim(xr.DataArray(buoyancy_glb_avg),dim='dim_0')
mld_glb_avg = mld_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
mld_glb_avg=detrend_dim(xr.DataArray(mld_glb_avg),dim='time')


infile='http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/ORAS5/1x1_grid/ileadfra/opa0'
data1 = xr.open_dataset(infile)
data1['time'] = pd.date_range("1979-01-15", periods=480, freq="M")

ice_clim = data1.ileadfra.groupby('time.month').mean(dim='time',skipna=True)
ice_anom = data1.ileadfra.groupby('time.month') - ice_clim

anomaly_ice = wgt_areaave(ice_anom, -90, -50, 0, 360)
ice_glb_avg = anomaly_ice / anomaly_ice.std()
ice_glb_avg=np.array(ice_glb_avg)
ice_glb_avg=ice_glb_avg[not_outlier1]
ice_glb_avg=xr.DataArray(ice_glb_avg)#detrend_dim(xr.DataArray(buoyancy_glb_avg),dim='dim_0')
ice_glb_avg = ice_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
ice_glb_avg=detrend_dim(xr.DataArray(ice_glb_avg),dim='time')




infile='http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/ORAS5/1x1_grid/sozotaux/opa0'
data1 = xr.open_dataset(infile)
data1['time'] = pd.date_range("1979-01-15", periods=480, freq="M")

infile='http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/ORAS5/1x1_grid/sometauy/opa0'
data2 = xr.open_dataset(infile)
data2['time'] = pd.date_range("1979-01-15", periods=480, freq="M")


def div_4pt_xr(U, V):
    """
    POP stencil operator for divergence
    using xarray
    """
    #U_at_lat_t = U + U.roll(lat=-1, roll_coords=False)  # avg U in y
    dUdx = U.roll(lon=-1, roll_coords=False) - U.roll(lon=1, roll_coords=False)  # dU/dx
    #V_at_lon_t = V + V.roll(lon=-1, roll_coords=False)  # avg V in x
    dVdy = V.roll(lat=-1, roll_coords=False) - V.roll(lat=1, roll_coords=False)  # dV/dy
    return dUdx,dVdy

rho0 = 1028


dx=(2*np.pi)/360
dy=(2*np.pi)/360


lat_wsc=data1.lat
lon_wsc=data1.lon

fy = 2. * 7.2921150e-5 * np.sin(np.deg2rad(data2['lat']))
fy = fy.where((np.abs(data2['lat']) > 3) & (np.abs(data2['lat']) < 87))  # Mask out the poles and equator regions

fx = 2. * 7.2921150e-5 * np.sin(np.deg2rad(data2['lat']))
fx = fx.where((np.abs(data2['lat']) > 3) & (np.abs(data2['lat']) < 87))  # Mask out the poles and equator regions

# Broadcast 'f' to match the dimensions of 'taux'
fx = fx.broadcast_like(data1.sozotaux, 'lat')
fy = fy.broadcast_like(data2.sometauy, 'lat')


def z_curl_xr(U, V, dx, dy, lat_wsc):
    """
    xr based
    """
    R = 6413 * (10 ** 3)
    dcos = np.cos(np.deg2rad(lat_wsc))  # Ensure positive value for cosine
    const = 1 / (R * dcos)
    const2 = 1 / (R * (dcos * dcos))
    vdy = 0.5 * V * dx * dcos
    udx = -0.5 * U * dy * dcos
    Udy, Vdx = div_4pt_xr(vdy, udx)
    zcurl = (const * Vdx + const2 * Udy) / (dx * dy)

    # Adjust sign in the southern hemisphere
    #southern_hemisphere = lat_wsc < 0
    #zcurl[southern_hemisphere] *= -1

    return zcurl, Udy, Vdx

rho0 = 1028


dx=(2*np.pi)/360
dy=(2*np.pi)/360

ekman_pumping, Udy, Vdx = z_curl_xr(data1.sozotaux, data2.sometauy , dx, dy, lat_wsc)


ekman_pumping_clim = ekman_pumping.groupby('time.month').mean(dim='time',skipna=True)
ekman_pumping_anom = ekman_pumping.groupby('time.month') - ekman_pumping_clim 

ekman_pumping_anom = wgt_areaave(ekman_pumping_anom, -90, -50, 0, 360)
ekman_pumping_glb_avg = ekman_pumping_anom / ekman_pumping_anom.std()
ekman_pumping_glb_avg=np.array(ekman_pumping_glb_avg)
ekman_pumping_glb_avg=ekman_pumping_glb_avg[not_outlier1]
ekman_pumping_glb_avg=xr.DataArray(ekman_pumping_glb_avg)#detrend_dim(xr.DataArray(buoyancy_glb_avg),dim='dim_0')
ekman_pumping_glb_avg = ekman_pumping_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
ekman_pumping_glb_avg=detrend_dim(xr.DataArray(ekman_pumping_glb_avg),dim='time')






infile='http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/ORAS5/1x1_grid/sosaline/opa0'
data1 = xr.open_dataset(infile)
data1['time'] = pd.date_range("1979-01-15", periods=480, freq="M")

infile='http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/ORAS5/1x1_grid/sosstsst/opa0'
data2 = xr.open_dataset(infile)
data2['time'] = pd.date_range("1979-01-15", periods=480, freq="M")

pt = xr.apply_ufunc(pdens, data1.sosaline, data2.sosstsst,
                    dask='parallelized',
                    output_dtypes=[data.salt.dtype])

rho_ref = 1035.
anom_density = pt - rho_ref

g = 9.81
buoyancy = -g * anom_density / rho_ref

buoyancy_clim = buoyancy.groupby('time.month').mean(dim='time',skipna=True)
buoyancy_anom = buoyancy.groupby('time.month') - buoyancy_clim 

anomaly_buoyancy = wgt_areaave(buoyancy_anom, -90, -50, -180, 180)
anomaly_buoyancy = anomaly_buoyancy / anomaly_buoyancy.std()
buoyancy_glb_avg=np.array(anomaly_buoyancy)
# Calculate z-scores and exclude outliers
mean = np.nanmean(buoyancy_glb_avg)
standard_deviation = np.nanstd(buoyancy_glb_avg)
distance_from_mean = abs(buoyancy_glb_avg - mean)
max_deviations = 3
not_outlier = distance_from_mean < max_deviations * standard_deviation
outlier=distance_from_mean > max_deviations * standard_deviation
buoyancy_glb_avg=buoyancy_glb_avg[not_outlier1]
buoyancy_glb_avg=xr.DataArray(buoyancy_glb_avg)#detrend_dim(xr.DataArray(buoyancy_glb_avg),dim='dim_0')
time=data.time.values[not_outlier1]
buoyancy_glb_avg = buoyancy_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
buoyancy_glb_avg=detrend_dim(xr.DataArray(buoyancy_glb_avg),dim='time')
# Plot the anomalies without outliers
plt.figure(figsize=(12, 6))
buoyancy_glb_avg.isel(time=slice(0,316)).coarsen(time=2).mean().plot()
plt.xlabel("Time", fontsize=20)
plt.ylabel("Buoyancy anomalies", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Buyoancy Anomalies (Deseasonalized, Detrended, and Excluding Outliers)", fontsize=22)
plt.tight_layout()
plt.show()


def pdens(S,theta):

    # --- Define constants (Table 1 Column 4, Wright 1997, J. Ocean Tech.)---
    a0 = 7.057924e-4
    a1 = 3.480336e-7
    a2 = -1.112733e-7

    b0 = 5.790749e8
    b1 = 3.516535e6
    b2 = -4.002714e4
    b3 = 2.084372e2
    b4 = 5.944068e5
    b5 = -9.643486e3

    c0 = 1.704853e5
    c1 = 7.904722e2
    c2 = -7.984422
    c3 = 5.140652e-2
    c4 = -2.302158e2
    c5 = -3.079464

    # To compute potential density keep pressure p = 100 kpa
    # S in standard salinity units psu, theta in DegC, p in pascals

    p = 100000.
    alpha0 = a0 + a1*theta + a2*S
    p0 = b0 + b1*theta + b2*theta**2 + b3*theta**3 + b4*S + b5*theta*S
    lambd = c0 + c1*theta + c2*theta**2 + c3*theta**3 + c4*S + c5*theta*S

    pot_dens = (p + p0)/(lambd + alpha0*(p + p0))

    return pot_dens





sosaline_clim = data1.sosaline.groupby('time.month').mean(dim='time',skipna=True)
sosaline_anom = data1.sosaline.groupby('time.month') - sosaline_clim 


anomaly_sosaline = wgt_areaave(sosaline_anom, -90, -50, -180, 180)
anomaly_sosaline = anomaly_sosaline / anomaly_sosaline.std()
sosaline_glb_avg=np.array(anomaly_sosaline)
# Calculate z-scores and exclude outliers
mean = np.nanmean(sosaline_glb_avg)
standard_deviation = np.nanstd(sosaline_glb_avg)
distance_from_mean = abs(sosaline_glb_avg - mean)
max_deviations = 3
not_outlier = distance_from_mean < max_deviations * standard_deviation
outlier=distance_from_mean > max_deviations * standard_deviation
sosaline_glb_avg=sosaline_glb_avg[not_outlier1]
sosaline_glb_avg=xr.DataArray(sosaline_glb_avg)#detrend_dim(xr.DataArray(buoyancy_glb_avg),dim='dim_0')
time=data.time.values[not_outlier1]
sosaline_glb_avg = sosaline_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
sosaline_glb_avg=detrend_dim(xr.DataArray(sosaline_glb_avg),dim='time')
# Plot the anomalies without outliers
plt.figure(figsize=(12, 6))
sosaline_glb_avg.isel(time=slice(0,316)).coarsen(time=2).mean().plot()
plt.xlabel("Time", fontsize=20)
plt.ylabel("Salinity anomalies", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Salinity Anomalies (Deseasonalized, Detrended, and Excluding Outliers)", fontsize=22)
plt.tight_layout()
plt.show()



sosstsst_clim = data2.sosstsst.groupby('time.month').mean(dim='time',skipna=True)
sosstsst_anom = data2.sosstsst.groupby('time.month') - sosstsst_clim 


anomaly_sosstsst = wgt_areaave(sosstsst_anom, -90, -50, -180, 180)
anomaly_sosstsst = anomaly_sosstsst / anomaly_sosstsst.std()
sosstsst_glb_avg=np.array(anomaly_sosstsst)
# Calculate z-scores and exclude outliers
mean = np.nanmean(sosstsst_glb_avg)
standard_deviation = np.nanstd(sosstsst_glb_avg)
distance_from_mean = abs(sosstsst_glb_avg - mean)
max_deviations = 3
not_outlier = distance_from_mean < max_deviations * standard_deviation
outlier=distance_from_mean > max_deviations * standard_deviation
sosstsst_glb_avg=sosstsst_glb_avg[not_outlier1]
sosstsst_glb_avg=xr.DataArray(sosstsst_glb_avg)#detrend_dim(xr.DataArray(buoyancy_glb_avg),dim='dim_0')
time=data.time.values[not_outlier1]
sosstsst_glb_avg = sosstsst_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
sosstsst_glb_avg=detrend_dim(xr.DataArray(sosstsst_glb_avg),dim='time')


pt = xr.apply_ufunc(pdens, data1.sosaline, data2.sosstsst,
                    dask='parallelized',
                    output_dtypes=[data1.sosaline.dtype])

pt=pt#(pt*area)/total_area

pd_clim = pt.groupby('time.month').mean(dim='time')
pd_anom = pt.groupby('time.month') - pd_clim



anomaly_pd = wgt_areaave(pd_anom, -90, -50, -180, 180)
anomaly_pd= anomaly_pd / anomaly_pd.std()
pd_glb_avg=np.array(anomaly_pd)
# Calculate z-scores and exclude outliers
mean = np.nanmean(pd_glb_avg)
standard_deviation = np.nanstd(pd_glb_avg)
distance_from_mean = abs(pd_glb_avg - mean)
max_deviations = 3
not_outlier = distance_from_mean < max_deviations * standard_deviation
outlier=distance_from_mean > max_deviations * standard_deviation
pd_glb_avg=pd_glb_avg[not_outlier1]
pd_glb_avg=xr.DataArray(pd_glb_avg)#detrend_dim(xr.DataArray(buoyancy_glb_avg),dim='dim_0')
time=data.time.values[not_outlier1]
pd_glb_avg = pd_glb_avg.rename({'dim_0': 'time'}).assign_coords({'time': time})
pd_glb_avg=detrend_dim(xr.DataArray(buoyancy_glb_avg),dim='time')
# Plot the anomalies without outliers
plt.figure(figsize=(12, 6))
pd_glb_avg.isel(time=slice(0,316)).coarsen(time=2).mean().plot()

    
# Example data (replace with your actual data)
sst = sosstsst_glb_avg#.coarsen(time=2).mean()
salinity = sosaline_glb_avg#.coarsen(time=2).mean()
buoyancy = buoyancy_glb_avg#.coarsen(time=2).mean()
ice = ice_glb_avg#.coarsen(time=2).mean()
ekman_pumping = ekman_pumping_glb_avg#.coarsen(time=2).mean()
ssh = ssh_glb_avg#.coarsen(time=2).mean()
no3 = no3_glb_avg#.coarsen(time=2).mean()
pt=pd_glb_avg
mld=mld_glb_avg

input_series = np.stack((sst, salinity, buoyancy, ekman_pumping, pt,mld), axis=1)
target_series = np.array(no3)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
import random
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import copy
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

seed =50 #2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)



# time series input
features =20
# training epochs
epochs =200 #1000
# synthetic time series dataset
ts_len = 1980
# test dataset size
test_len =106
# temporal casual layer channels
channel_sizes = [9] * 3
# convolution kernel size
kernel_size =3 #5
dropout = 0.1


#ts = generate_time_series(ts_len)
train_ratio=0.7
#ts_diff_y = ts_diff(ts[:, 0])
#ts_diff = copy.deepcopy(ts)
#ts_diff[:, 0] = ts_diff_y
l1_factor=3*10^(-10)
l2_factor=3*10^(-10)

C=[0,1,2]
lr2=[0.0001,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001,0.0002,0.0003]



#E=np.concatenate((X_no3[:,np.newaxis],X_po4[:,np.newaxis]),axis=1)
#E=[X_no3[:,np.newaxis],X_po4[:,np.newaxis]]
#print("E",E.shape)



N=24

prediction1=np.zeros((4,24))
prediction2=np.zeros((4,24))
prediction3=np.zeros((4,24))
prediction4=np.zeros((4,24))
prediction5=np.zeros((4,24))
prediction6=np.zeros((4,24))
prediction7=np.zeros((4,24))
prediction8=np.zeros((4,24))
prediction9=np.zeros((4,24))
prediction10=np.zeros((4,24))



#print("X_no3",X_no3.shape)
#print("X_npp",X_npp.shape)


#npp_index=npp_index.T
#print("npp_index_GFDL",npp_index.shape)



k=0
#M=[3,6,9,12,15,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168,174,180,186,192,198,204,210,216,222,228,234,240,246,252,258,264]

#M=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180,183,186,189,192,195,198,201,204,207,210,213,216,219,222,225,228,231,234,237,240,243,246,249,252,255,258,261,264]
M=np.arange(1,24)

years=np.arange(1850,2015,1/12)

class EarlyStopping:

    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt',trace_func=print):

        #Args:
        """
                            Default: False
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.val_loss_min = np.Inf
        self.trace_func = trace_func

        
    def __call__(self, val_loss, model):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        energy = torch.tanh(self.W(x))  # shape: (batch_size, seq_length, hidden_size)
        energy = energy.view(x.size(0), -1, self.num_heads, self.head_size)  # shape: (batch_size, seq_length, num_heads, head_size)
        energy = energy.permute(0, 2, 1, 3)  # shape: (batch_size, num_heads, seq_length, head_size)

        attention_weights = torch.softmax(self.V(energy), dim=2)  # shape: (batch_size, num_heads, seq_length, 1)
        context_vector = torch.sum(attention_weights * energy, dim=2)  # shape: (batch_size, num_heads, head_size)

        context_vector = context_vector.view(x.size(0), -1)  # shape: (batch_size, hidden_size)
        return context_vector


class TemporalCasualLayer(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout ):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride':      1,
            'padding':     padding,
            'dilation':    dilation
        }

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        self.crop1 = Crop(padding)
        self.relu1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)
        
        self.conv3 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop3 = Crop(padding)
        self.relu3 = nn.Tanh()
        self.dropout3 = nn.Dropout(dropout)
        
        self.conv4 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop4 = Crop(padding)
        self.relu4 = nn.Tanh()
        self.dropout4 = nn.Dropout(dropout)

        self.conv5 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop5 = Crop(padding)
        self.relu5 = nn.Tanh()
        self.dropout5 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2,
                                 self.conv3, self.crop3, self.relu3, self.dropout3
                                
                                 )
        self.residual = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.residual is not None:
            self.residual.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        y = self.net(x)

        output = self.relu(y + residual)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        energy = torch.tanh(self.W(x))  # shape: (batch_size, seq_length, hidden_size)
        energy = energy.view(x.size(0), -1, self.num_heads, self.head_size)  # shape: (batch_size, seq_length, num_heads, head_size)
        energy = energy.permute(0, 2, 1, 3)  # shape: (batch_size, num_heads, seq_length, head_size)

        attention_weights = torch.softmax(self.V(energy), dim=2)  # shape: (batch_size, num_heads, seq_length, 1)
        context_vector = torch.sum(attention_weights * energy, dim=2)  # shape: (batch_size, num_heads, head_size)

        context_vector = context_vector.view(x.size(0), -1)  # shape: (batch_size, hidden_size)
        return context_vector

class Crop(nn.Module):

    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()
    

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, in_channels, T) where N is the batch_size, and T is the sequence length
        mask = np.array([[1 if i>j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        if input.is_cuda:
            mask = torch.ByteTensor(mask).cuda(input.get_device())
        else:
            mask = torch.ByteTensor(mask)
        # mask = mask.bool()
        
        input = input.permute(0,2,1) # input: [N, T, inchannels]
        keys = self.linear_keys(input) # keys: (N, T, key_size)
        query = self.linear_query(input) # query: (N, T, key_size)
        values = self.linear_values(input) # values: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))

        weight_temp = F.softmax(temp / self.sqrt_key_size, dim=1) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_vert = F.softmax(temp / self.sqrt_key_size, dim=1) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_hori = F.softmax(temp / self.sqrt_key_size, dim=2) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp = (weight_temp_hori + weight_temp_vert)/2
        value_attentioned = torch.bmm(weight_temp, values).permute(0,2,1) # shape: (N, T, value_size)
       
        return value_attentioned, weight_temp # value_attentioned: [N, in_channels, T], weight_temp: [N, T, T]
    

class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.1):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_params = {
            'kernel_size': kernel_size,
            'stride': 1,
            'dropout': dropout
        }

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            tcl_params['dilation'] = dilation
            tcl = TemporalCasualLayer(in_channels, out_channels, **tcl_params)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(SparseMultiHeadAttention, self).__init__()
        assert input_size % num_heads == 0, "Input size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_size = input_size // num_heads

        self.W_query = nn.Linear(input_size, input_size)
        self.W_key = nn.Linear(input_size, input_size)
        self.W_value = nn.Linear(input_size, input_size)

    def forward(self, x):
        batch_size, seq_length, input_size = x.size()
        
        queries = self.W_query(x)  # shape: (batch_size, seq_length, input_size)
        keys = self.W_key(x)  # shape: (batch_size, seq_length, input_size)
        values = self.W_value(x)  # shape: (batch_size, seq_length, input_size)

        # Reshape queries, keys, and values for multi-head attention
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        # Perform attention mechanism
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_size)  # shape: (batch_size, num_heads, seq_length, seq_length)
        attention_weights = torch.softmax(energy, dim=-1)  # shape: (batch_size, num_heads, seq_length, seq_length)
        context_vector = torch.matmul(attention_weights, values)  # shape: (batch_size, num_heads, seq_length, head_size)

        # Rearrange context_vector to match the original shape
        context_vector = context_vector.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, input_size)

        return context_vector   
    
class TCNN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout,l1_factor,l2_factor,num_heads):
        super(TCNN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.conv1 = TemporalConvolutionNetwork(input_channels, [32, 64, 128], kernel_size, dropout)
        self.attention = SparseMultiHeadAttention(num_channels[-1], num_heads)
        #self.attention = MultiHeadAttention(num_channels[-1], attention_hidden_size, num_heads)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.dropout = nn.Dropout(dropout)
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #x_mag = self.fourier_feature_engineering(x)
        y = self.tcn(x)
        y = self.dropout(y)

        # L1 Regularization
        l1_reg = torch.tensor(0.001)
        if self.l1_factor > 0:
            for param in self.parameters():
                l1_reg += torch.norm(param, p=1)

        # L2 Regularization
        l2_reg = torch.tensor(0.001)
        if self.l2_factor > 0:
            for param in self.parameters():
                l2_reg += torch.norm(param, p=2)

        #attended_features = self.attention(y[:, :, -1])
        y = y.transpose(1, 2)
        attended_features = self.attention(y)
        attended_features = attended_features.transpose(1, 2)  # Restore original shape
        out = self.linear(attended_features[:, :, -1])

        if self.l1_factor > 0:
            out += self.l1_factor * l1_reg

        if self.l2_factor > 0:
            out += 0.5 * self.l2_factor * l2_reg

        return out  
      

from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import copy


def train_model(model, X, y, epochs, optimizer, mse_loss, early_stopping, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params = None
    min_val_loss = np.inf

    all_training_losses = []
    all_validation_losses = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        x_train, x_val = torch.tensor(X_train).float(), torch.tensor(X_val).float()
        y_train, y_val = torch.tensor(y_train).float(), torch.tensor(y_val).float()

        fold_training_losses = []
        fold_validation_losses = []
        fold_min_val_loss = np.inf

        for t in range(epochs):
            model.train()
            optimizer.zero_grad()

            prediction = model(x_train)
            loss = mse_loss(prediction, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_prediction = model(x_val)
                val_loss = mse_loss(val_prediction, y_val)

            fold_training_losses.append(loss.item())
            fold_validation_losses.append(val_loss.item())

            if val_loss.item() < fold_min_val_loss:
                best_fold_params = copy.deepcopy(model.state_dict())
                fold_min_val_loss = val_loss.item()
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {t} in fold {fold}")
                break

            if t % 10 == 0:
                print(f'Fold {fold}, Epoch {t}. Train Loss: {round(loss.item(), 4)}, Val Loss: {round(val_loss.item(), 4)}')

        all_training_losses.append(fold_training_losses)
        all_validation_losses.append(fold_validation_losses)

        if fold_min_val_loss < min_val_loss:
            best_params = best_fold_params
            min_val_loss = fold_min_val_loss

    return best_params, all_training_losses, all_validation_losses, min_val_loss




def ts_diff(ts):
    diff_ts = [0] * len(ts)
    for i in range(1, len(ts)):
        diff_ts[i] = ts[i] - ts[i - 1]
    return diff_ts


input2=[48]
output2=[48]
skill_so=np.zeros((4,9))



seed2=list(range(4))

for ii in range(4):
    seed =seed2[ii] #50
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    w=0
    i=0
    Z=[0,12,24,36,48]
    #D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
    D2=np.zeros((264,6))
    for j in range(6):
        D2[:,j]=ts_diff(input_series[:,j])
    #T2[:,]=target_series[:,]
    T2=np.zeros((264,))
    for j in range(1):
        T2[:,]=ts_diff(target_series[:,])
    # Use the mask to exclude NaN value
    
    #D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
    #T2=target_series#np.concatenate((no3_index[:,np.newaxis],tos_index[:,np.newaxis]),axis=1)
#   D=np.concatenate((X_no3[:-Z[i],np.newaxis],X_po4[:-Z[i],np.newaxis],pc1_tos[:-Z[i],:10],pc1_so[:-Z[i],:5],pc1_zos[:-Z[i],:5]),axis=1) 
    X_ss, Y_mm =  split_sequences(D2,T2[:,], input2[0],output2[0])
    print("X_ss",X_ss.shape)
    print("y_mm",Y_mm.shape)
    train_ratio=0.7
    train_len = round(len(X_ss[:-(input2[0]+output2[0]+24)]) * train_ratio)
    test_len=input2[0]+output2[0] #150/3
    X_train, Y_train= X_ss[:-(input2[0]+output2[0]+24)],\
                                   Y_mm[:-(input2[0]+output2[0]+24)],\
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]

    print("X_train",X_train.shape)
    X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]
    x_train = torch.tensor(data = X_train).float()
    y_train = torch.tensor(data = Y_train).float()

    x_val = torch.tensor(data = X_val).float()
    y_val = torch.tensor(data = Y_val).float()

    #x_test = torch.tensor(data = X_test).float()
    #y_test = torch.tensor(data = Y_test).float()
    x_train = x_train.transpose(1, 2)
    x_val = x_val.transpose(1, 2)
    #x_test = x_test.transpose(1, 2)

    #y_train = y_train[:, :, 0]
    #y_val = y_val[:,:,0]
    print("x_train",x_train.shape)
    print("y_train",y_train.shape)
    train_len = x_train.size()[0]

    model_params = {
    'input_size': D2.shape[1], #60
    'output_size':  48,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout,
    'l1_factor':l1_factor,
    'l2_factor':l2_factor,
    'num_heads':3
    }
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNN(**model_params)

    best_params = None
    min_val_loss = sys.maxsize

    training_loss = []
    validation_loss = []

    #model = model.to(device)
    n_splits = 3  # Number of fold

    # Data preparation
    X_ss, Y_mm = split_sequences(D2,target_series[:,], input2[0], output2[0])
    print("X_ss", X_ss.shape)
    print("y_mm", Y_mm.shape)

    # Results storage
    all_train_loss = []
    all_val_loss = []
    #early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Train the model on the current fold
    kernel_sizes = [3,4,5,6]
    channel_sizes_options = [[6,6,6],[9,9,9],[12,12,12],[18,18,18]]
    dropout_list=[0,0.1,0.2,0.3]
    epochs = 200

    best_config = None
    best_val_loss = np.inf
    best_model_params = None

    train_losses=[]
    val_losses=[]
    best_model_path = 'best_model.pth'
    
    for kernel_size in kernel_sizes:
      for channel_sizes in channel_sizes_options:
         for dropout in dropout_list:

            print(f"Testing Kernel Size: {kernel_size}, Channel Sizes: {channel_sizes}")

            model_params = {
            'input_size': 6,  # Adjust this as needed
            'output_size': 48,
            'num_channels': channel_sizes,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'l1_factor': l1_factor,
            'l2_factor': l2_factor,
            'num_heads': 3
            }

            model = TCNN(**model_params)
            optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay=0.000001, lr=0.007) 
            mse_loss = nn.MSELoss()

            early_stopping = EarlyStopping(patience=10, verbose=True)

            # Train the model and get the validation loss
            best_params, train_loss, val_loss, val_loss_min = train_model(
            model, x_train, y_train, epochs, optimizer, mse_loss,early_stopping
            )

            if val_loss_min < best_val_loss:
               best_val_loss = val_loss_min
               best_config = (kernel_size, channel_sizes, dropout)
               best_model_params = best_params

            train_losses.append(train_loss)
            val_losses.append(val_loss)




    # Optionally load the best model parameters
    model.load_state_dict(best_params)
    
    avg_train_loss = np.mean([np.min(losses) for losses in train_loss])
    avg_val_loss = np.mean([np.min(losses) for losses in val_loss])
    plt.figure()
    plt.title('Training Progress')
    plt.yscale("log")
    plt.plot(avg_train_loss, label = 'train')
    plt.plot(avg_val_loss, label = 'validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    plt.savefig("loss_"+str(w)+".png")


    def ts_int(ts_diff, ts_base, start=0):
    #    """
    #    Integrate a differenced time series using cumulative sum.

    #    Parameters:
    #    - ts_diff (numpy array): The differenced time series.
    #    - ts_base (numpy array): The base time series.
    #    - start (float): The initial value for integration.

    #    Returns:
    #    - ts_integrated (numpy array): The integrated time series.
    #    """
        ts_diff = np.asarray(ts_diff)
        ts_base = np.asarray(ts_base)

        ts_integrated = np.empty_like(ts_diff)
        ts_integrated[0] = start + ts_diff[0]

        # Use cumulative sum for integration
        ts_integrated[1:] = np.cumsum(ts_diff[1:]) + ts_base[:-1]
        return ts_integrated.tolist()
    
    #def ts_int(ts_diff, ts_base, start):
#        ts_int = [start]
#        for i in range(1, len(ts_diff)):
#            ts_int.append(ts_int[i-1] + ts_diff[i-1] + ts_base[i-1])
#        return np.array(ts_int)

    for N in range(24):
        if N==0:
           test_len=input2[0]+output2[0]
           X_test, Y_test= X_ss[-48:],Y_mm[-48:]
           x_test = torch.tensor(data = X_test).float()
           y_test = torch.tensor(data = Y_test).float()
           x_test = x_test.transpose(1, 2)
           #y_test=y_test[:,:,ii]

           best_model = TCNN(
           input_size=6,
           output_size=48,
           num_channels=best_config[1],
           kernel_size=best_config[0],
           dropout=best_config[2],
           l1_factor=l1_factor,
           l2_factor=l2_factor,
           num_heads=3
           )
           best_model.eval()
           #best_model.load_state_dict(best_model_params)

           tcn_prediction = best_model(x_test)

           #print('tcn_prediction',tcn_prediction[-1,:].detach().numpy().shape)
           A=0
           years=np.arange(1996-int(A/12),2015,1/12)
           test_len=input2[0]+output2[0]  #150/3

           Z=ts_int(
            tcn_prediction[-1,:].tolist(),
            target_series[-48:,],
            start =  target_series[-48-1,]
            )
           #Q=["NO3 anomaly","PO4 anomaly","first pc of SST"]
           #ci = 0.1 * np.std(Z[input2[0]:]) / np.mean(Z[input2[0]:])
           #95% confidence interval
           #plt.figure()
           #plt.fill_between(years[-108:], (Z[input2[0]:]-ci), (Z[input2[0]:]+ci), color='green', alpha=0.5)
           #plt.plot(years[-108:],Z[input2[0]:],label = 'tcn',color='k',linewidth=2.5)
           #plt.plot(years[-108:],T2[-108:], label = 'real',color='r',linewidth=2.5)
           #plt.ylabel(Q[ii],fontsize=13)
           #plt.xlabel("Years",fontsize=13)
           #plt.xticks(fontsize=13)
           #plt.yticks(fontsize=13)
           #plt.legend()
           #plt.show()
           #plt.savefig('forecast_TCNN_GFDL_0.png')

           
           prediction1[ii,N]=Z[-1]
           prediction2[ii,N]=Z[-6]
           prediction3[ii,N]=Z[-12]
           prediction4[ii,N]=Z[-18]
           prediction5[ii,N]=Z[-24]
           prediction6[ii,N]=Z[-30]
           prediction7[ii,N]=Z[-36]
           prediction8[ii,N]=Z[-42] 
           


        if N>0:
           X_test, Y_test= X_ss[-48-N:-N],Y_mm[-48-N:-N]
           x_test = torch.tensor(data = X_test).float()
           y_test = torch.tensor(data = Y_test).float()
           x_test = x_test.transpose(1, 2)
           
           best_model = TCNN(
           input_size=6,
           output_size=48,
           num_channels=best_config[1],
           kernel_size=best_config[0],
           dropout=best_config[2],
           l1_factor=l1_factor,
           l2_factor=l2_factor,
           num_heads=3
           )
           best_model.eval()
           #best_model.load_state_dict(best_model_params)

           tcn_prediction = best_model(x_test)


           A=24
           test_len=input2[0]+output2[0]  #150/3
           
           Z=ts_int(
            tcn_prediction[-1,:].tolist(),
            target_series[-48-N:-N,],
            start = target_series[-48-N-1,]
            )
           Q=["NO3 anomaly","PO4 anomaly","first pc of SST"]
           ci = 0.1 * np.std(Z[input2[0]:]) / np.mean(Z[input2[0]:])
           #95% confidence interval
           #plt.figure()
           #plt.fill_between(years[-96-N:-N],(Z[20:]-ci), (Z[20:]+ci), color='green', alpha=0.5)
           #plt.plot(years[-108-N:-N],Z[input2[0]:],label = 'tcn',color='k',linewidth=2.5)
           #plt.plot(years[-108-N:-N],T2[-108-N:-N,], label = 'real',color='r',linewidth=2.5)
           #plt.ylabel(Q[ii],fontsize=13)
           #plt.xlabel("Years",fontsize=13)
           #plt.xticks(fontsize=13)
           #plt.yticks(fontsize=13)
           #plt.legend()
           #plt.show( 
              
           prediction1[ii,N]=Z[-1]
           prediction2[ii,N]=Z[-6]
           prediction3[ii,N]=Z[-12]
           prediction4[ii,N]=Z[-18]
           prediction5[ii,N]=Z[-24]
           prediction6[ii,N]=Z[-30]
           prediction7[ii,N]=Z[-36]
           prediction8[ii,N]=Z[-42] 

    Q=6
    A=np.corrcoef(prediction1[ii,::-1],target_series[-24:,])
    skill_so[ii,8]=A[1][0]
    A=np.corrcoef(prediction2[ii,::-1],target_series[-24-Q:-6,])
    skill_so[ii,7]=A[1][0]
    A=np.corrcoef(prediction3[ii,::-1],target_series[-24-Q*2:-12,])
    skill_so[ii,6]=A[1][0]
    A=np.corrcoef(prediction4[ii,::-1],target_series[-24-Q*3:-18,])
    skill_so[ii,5]=A[1][0]
    A=np.corrcoef(prediction5[ii,::-1],target_series[-24-Q*4:-24,])
    skill_so[ii,4]=A[1][0]
    A=np.corrcoef(prediction6[ii,::-1],target_series[-24-Q*5:-30,])
    skill_so[ii,3]=A[1][0]
    A=np.corrcoef(prediction7[ii,::-1],target_series[-24-Q*6:-36,])
    skill_so[ii,2]=A[1][0]
    A=np.corrcoef(prediction8[ii,::-1],target_series[-24-Q*7:-42,])
    skill_so[ii,1]=A[1][0]
    skill_so[ii,0]=1
    
plt.figure(figsize=(10, 10),dpi=1200) 
plt.plot(np.mean(skill_so,axis=0),'r')
plt.ylabel("skill (correlation values)",fontsize=13)
plt.xlabel("Time (years)",fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()

# Example data (replace with your actual data)
sst = sosstsst_glb_avg#.coarsen(time=2).mean()
salinity = sosaline_glb_avg#.coarsen(time=2).mean()
buoyancy = buoyancy_glb_avg#.coarsen(time=2).mean()
ice = ice_glb_avg#.coarsen(time=2).mean()
ekman_pumping = ekman_pumping_glb_avg#.coarsen(time=2).mean()
ssh = ssh_glb_avg#.coarsen(time=2).mean()
no3 = no3_glb_avg#.coarsen(time=2).mean()
pt=pd_glb_avg
mld=mld_glb_avg

input_series = np.stack((sst, salinity, buoyancy, ekman_pumping, pt,mld), axis=1)
target_series = np.array(no3)

from sklearn.linear_model import LinearRegression
model =LinearRegression()#LinearRegression()
alpha = 0.01  # Regularization strength (adjus



# Use the mask to exclude NaN value
input2=[48]
output2=[48]
#D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
T2=ts_diff(target_series)

D2=np.zeros((264,6))
for j in range(6):
    D2[:,j]=ts_diff(input_series[:,j])

    
X_ss, Y_mm =  split_sequences(D2,T2, input2[0],output2[0])
print("X_ss",X_ss.shape)
print("y_mm",Y_mm.shape)
train_ratio=0.8
train_len = round(len(X_ss[:-(input2[0]+output2[0]+24)]) * train_ratio)
test_len=input2[0]+output2[0] #150/3
X_train, Y_train= X_ss[:-(input2[0]+output2[0]+24)],\
                                   Y_mm[:-(input2[0]+output2[0]+24)],\
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]


print("X_train",X_train.shape)
X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]

x_train = torch.tensor(data = X_train).float()
y_train = torch.tensor(data = Y_train).float()
x_val = torch.tensor(data = X_val).float()
y_val = torch.tensor(data = Y_val).float()
X_train = X_train.reshape(X_train.shape[0], -1)

X_test,y_test=X_ss[-(48+48+24):],Y_mm[-(48+48+24):]
X_test = X_test.reshape(X_test.shape[0], -1)

model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)




prediction1=np.zeros((24))
prediction2=np.zeros((24))
prediction3=np.zeros((24))
prediction4=np.zeros((24))
prediction5=np.zeros((24))
prediction6=np.zeros((24))
prediction7=np.zeros((24))
prediction8=np.zeros((24))
prediction9=np.zeros((24))

skill_linreg=np.zeros((9))
for N in range(24):
  if N==0:

     X_test, Y_test= X_ss[-36:],Y_mm[-36:]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-48:,],
           start = target_series[-48-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-12]
     prediction3[N]=Z[-12]
     prediction4[N]=Z[-18]
     prediction5[N]=Z[-24]
     prediction6[N]=Z[-30]
     prediction7[N]=Z[-36]
     prediction8[N]=Z[-42] 
     
  if N>0:

     X_test, Y_test= X_ss[-48-N:-N],Y_mm[-48-N:-N]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-48-N:-N,],
           start = target_series[-48-N-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-6]
     prediction3[N]=Z[-12]
     prediction4[N]=Z[-18]
     prediction5[N]=Z[-24]
     prediction6[N]=Z[-30]
     prediction7[N]=Z[-36]
     prediction8[N]=Z[-42] 


Q=6
A=np.corrcoef(prediction1[::-1],target_series[-24:,])
skill_linreg[8]=A[1][0]
A=np.corrcoef(prediction2[::-1],target_series[-24-Q:-6,])
skill_linreg[7]=A[1][0]
A=np.corrcoef(prediction3[::-1],target_series[-24-Q*2:-12,])
skill_linreg[6]=A[1][0]
A=np.corrcoef(prediction4[::-1],target_series[-24-Q*3:-18,])
skill_linreg[5]=A[1][0]
A=np.corrcoef(prediction5[::-1],target_series[-24-Q*4:-24,])
skill_linreg[4]=A[1][0]
A=np.corrcoef(prediction6[::-1],target_series[-24-Q*5:-30,])
skill_linreg[3]=A[1][0]
A=np.corrcoef(prediction7[::-1],target_series[-24-Q*6:-36,])
skill_linreg[2]=A[1][0]
A=np.corrcoef(prediction8[::-1],target_series[-24-Q*7:-42,])
skill_linreg[1]=A[1][0]
skill_linreg[0]=1   






from sklearn.linear_model import Ridge
alpha = 0.00005  # Regularization strength (adjust as needed)
model = Ridge(alpha=alpha)

D2=np.zeros((264,6))
for j in range(6):
    D2[:,j]=ts_diff(input_series[:,j])


T2=ts_diff(target_series[:,])
# Use the mask to exclude NaN value
input2=[48]
output2=[48]
#D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
#T2=target_series
X_ss, Y_mm =  split_sequences(D2,T2, input2[0],output2[0])
print("X_ss",X_ss.shape)
print("y_mm",Y_mm.shape)
train_ratio=0.8
train_len = round(len(X_ss[:-(input2[0]+output2[0]+24)]) * train_ratio)
test_len=input2[0]+output2[0] #150/3
X_train, Y_train= X_ss[:-(input2[0]+output2[0]+24)],\
                                   Y_mm[:-(input2[0]+output2[0]+24)],\
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]


print("X_train",X_train.shape)
X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]

x_train = torch.tensor(data = X_train).float()
y_train = torch.tensor(data = Y_train).float()

x_val = torch.tensor(data = X_val).float()
y_val = torch.tensor(data = Y_val).float()
X_train = X_train.reshape(X_train.shape[0], -1)

X_test,y_test=X_ss[-(48+48+24):],Y_mm[-(48+48+24):]
X_test = X_test.reshape(X_test.shape[0], -1)

model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)




prediction1=np.zeros((24))
prediction2=np.zeros((24))
prediction3=np.zeros((24))
prediction4=np.zeros((24))
prediction5=np.zeros((24))
prediction6=np.zeros((24))
prediction7=np.zeros((24))
prediction8=np.zeros((24))
prediction9=np.zeros((24))


skill_ridge=np.zeros((9))
for N in range(24):
  if N==0:

     X_test, Y_test= X_ss[-48:],Y_mm[-48:]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-48:,],
           start = target_series[-48-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-6]
     prediction3[N]=Z[-12]
     prediction4[N]=Z[-18]
     prediction5[N]=Z[-24]
     prediction6[N]=Z[-30]
     prediction7[N]=Z[-36]
     prediction8[N]=Z[-42] 
     
  if N>0:

     X_test, Y_test= X_ss[-48-N:-N],Y_mm[-48-N:-N]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-48-N:-N,],
           start = target_series[-48-N-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-6]
     prediction3[N]=Z[-12]
     prediction4[N]=Z[-18]
     prediction5[N]=Z[-24]
     prediction6[N]=Z[-30]
     prediction7[N]=Z[-36]
     prediction8[N]=Z[-42] 
   


Q=6
A=np.corrcoef(prediction1[::-1],target_series[-24:,])
skill_ridge[8]=A[1][0]
A=np.corrcoef(prediction2[::-1],target_series[-24-Q:-6,])
skill_ridge[7]=A[1][0]
A=np.corrcoef(prediction3[::-1],target_series[-24-Q*2:-12,])
skill_ridge[6]=A[1][0]
A=np.corrcoef(prediction4[::-1],target_series[-24-Q*3:-18,])
skill_ridge[5]=A[1][0]
A=np.corrcoef(prediction5[::-1],target_series[-24-Q*4:-24,])
skill_ridge[4]=A[1][0]
A=np.corrcoef(prediction6[::-1],target_series[-24-Q*5:-30,])
skill_ridge[3]=A[1][0]
A=np.corrcoef(prediction7[::-1],target_series[-24-Q*6:-36,])
skill_ridge[2]=A[1][0]
A=np.corrcoef(prediction8[::-1],target_series[-24-Q*7:-42,])
skill_ridge[1]=A[1][0]
skill_ridge[0]=1 



lags = range(25)
persistence=np.zeros((1,25))
import statsmodels.api as sm
acorr = sm.tsa.acf(target_series, nlags = len(lags)-1)
#auto2[im,:]=acorr
#acorr = sm.tsa.acf(savitzky_golay(A2[:,im],12,3), nlags = len(lags)-1)
persistence[0,:]=acorr



import numpy as np
import matplotlib.pyplot as plt

# Assuming skill_so, skill_linreg, skill_ridge, persistence are defined arrays

# Calculate standard deviation
ci = np.std(skill_so, axis=0)

# Plotting
plt.figure()

plt.fill_between(range(len(np.mean(skill_so, axis=0))),
                 np.mean(skill_so, axis=0) - ci,
                 np.mean(skill_so, axis=0) + ci,
                 color='green', alpha=0.5, label='Skill_so Spread')
plt.plot(np.mean(skill_so, axis=0), 'r', label='Mean skill_so')
plt.plot(skill_linreg, 'g', label='Skill_linreg')
plt.plot(skill_ridge, 'k', label='Skill_ridge')
plt.plot(persistence[0, ::3], 'b', label='Persistence')
plt.ylabel("Skill", fontsize=13)
plt.xlabel("Time (6-month)", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=6)
plt.grid(True)
plt.show()



# now do the case with the GFDL-ESM4 model

import intake

cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(cat_url)
col
cat = col.search(experiment_id=['historical'], variable_id=['tos','so','no3','thetao','mlotst'],source_id=['GFDL-ESM4'],table_id=['Omon'], grid_label=['gr'])
dset_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
dset_dict.keys()

ds = dset_dict['CMIP.NOAA-GFDL.GFDL-ESM4.historical.Omon.gr']

min_lon = 0
min_lat = -90
min_depth = 0

max_lon = 360
max_lat = -50
max_depth = 50

mask_lat = (ds.lat >= min_lat) & (ds.lat <= max_lat)
mask_depth = (ds.lev >= min_depth) & (ds.lev <= max_depth)
ds = ds.where(mask_lat & mask_depth, drop=True)
ds=ds.mean('lev').mean('dcpp_init_year').mean('member_id')

import xesmf as xe
infile = '/Users/navarra/Desktop/article/tauuo-GFDL-ESM4.nc'
data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data['time'] = pd.date_range("1850-01-01", periods=1980, freq="M")
#data=data.isel(time=slice(120, 768))



taux=data.tauuo
lon_wsc=np.array(taux.x)
lat_wsc=np.array(taux.y)


infile = '/Users/navarra/Desktop/article/tauvo-GFDL-ESM4.nc'
data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data['time'] = pd.date_range("1850-01-01", periods=1980, freq="M")
#data=data.isel(time=slice(120, 768))

tauy=data.tauvo



grid_1x1=xe.util.grid_global(1,1)

grid_1x1['y']=grid_1x1['y']-90
grid_1x1['y_b']=grid_1x1['y_b']-90
grid_1x1['x']=grid_1x1['x']-180
grid_1x1['x_b']=grid_1x1['x_b']-180

regrid_to_1x1 = xe.Regridder(taux, grid_1x1, 'bilinear', periodic=True)

taux2 = regrid_to_1x1(taux, keep_attrs=False)



regrid_to_1x1 = xe.Regridder(tauy, grid_1x1, 'bilinear', periodic=True)

tauy2 = regrid_to_1x1(tauy, keep_attrs=False)

new_vector = xr.DataArray(np.ones((1980, 180, 360)),
                           dims={'time': 1980, 'lat': 180, 'lon': 360},
                           coords={'time': np.arange(1980), 'lat': np.arange(-90, 90), 'lon': np.arange(0, 360)})

taux3 = new_vector.copy(data=taux2.data)

tauy3 = new_vector.copy(data=tauy2.data)
lat_wsc=tauy3.lat
lon_wsc=taux3.lon


fy = 2. * 7.2921150e-5 * np.sin(np.deg2rad(tauy3['lat']))
fy = fy.where((np.abs(tauy3['lat']) > 3) & (np.abs(tauy3['lat']) < 87))  # Mask out the poles and equator regions

fx = 2. * 7.2921150e-5 * np.sin(np.deg2rad(taux3['lat']))
fx = fx.where((np.abs(tauy3['lat']) > 3) & (np.abs(taux3['lat']) < 87))  # Mask out the poles and equator regions

# Broadcast 'f' to match the dimensions of 'taux'
fx = fx.broadcast_like(taux3, 'lat')
fy = fy.broadcast_like(tauy3, 'lat')

def div_4pt_xr(U, V):
    """
    POP stencil operator for divergence
    using xarray
    """
    #U_at_lat_t = U + U.roll(lat=-1, roll_coords=False)  # avg U in y
    dUdx = U.roll(lon=-1, roll_coords=False) - U.roll(lon=1, roll_coords=False)  # dU/dx
    #V_at_lon_t = V + V.roll(lon=-1, roll_coords=False)  # avg V in x
    dVdy = V.roll(lat=-1, roll_coords=False) - V.roll(lat=1, roll_coords=False)  # dV/dy
    return dUdx,dVdy


dx=(2*np.pi)/360
dy=(2*np.pi)/360

def z_curl_xr(U, V, dx, dy, lat_wsc):
    """
    xr based
    """
    R = 6413 * (10 ** 3)
    dcos = np.cos(np.deg2rad(lat_wsc))  # Ensure positive value for cosine
    const = 1 / (R * dcos)
    const2 = 1 / (R * (dcos * dcos))
    vdy = 0.5 * V * dx * dcos
    udx = -0.5 * U * dy * dcos
    Udy, Vdx = div_4pt_xr(vdy, udx)
    zcurl = (const * Vdx + const2 * Udy) / (dx * dy)

    # Adjust sign in the southern hemisphere
    #southern_hemisphere = lat_wsc < 0
    #zcurl[southern_hemisphere] *= -1

    return zcurl, Udy, Vdx

ekman_pumping, Udy, Vdx = z_curl_xr(taux3 / (rho0 * fx), tauy3 / (rho0 * fy), dx, dy, lat_wsc)
ekman_pumping=ekman_pumping.sel(lat=slice(-90,-50))
ekman_pumping=ekman_pumping.transpose('time','lat','lon')
ekman_pumping['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
ekman_pumping=ekman_pumping.isel(time=slice(1980-444,1980))

ekman_pumping_clim = ekman_pumping.groupby('time.month').mean(dim='time',skipna=True)
ekman_pumping_anom = ekman_pumping.groupby('time.month') - ekman_pumping_clim 

ekman_pumping_anom = wgt_areaave(ekman_pumping_anom, -90, -50, 0, 360)
ekman_pumping_anom=detrend_dim(ekman_pumping_anom,dim='time')

ds['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
ds=ds.isel(time=slice(1980-444,1980))

so_detrend=ds.so
so_clim = so_detrend.groupby('time.month').mean(dim='time',skipna=True)
so_anom = so_detrend.groupby('time.month') - so_clim
#sst_anom=sst_anom.coarsen(time=3).mean()
so_anom=wgt_areaave(so_anom,-90,-50,0,360)
so_anom=detrend_dim(so_anom,dim='time')
so_index=so_anom/so_anom.std()

tos_detrend=ds.tos
tos_clim = tos_detrend.groupby('time.month').mean(dim='time',skipna=True)
tos_anom = tos_detrend.groupby('time.month') - tos_clim
#sst_anom=sst_anom.coarsen(time=3).mean()
tos_anom=wgt_areaave(tos_anom,-90,-50,0,360)
tos_anom=detrend_dim(tos_anom,dim='time')
tos_index=tos_anom/tos_anom.std()

mld_detrend=ds.mlotst
mld_clim = mld_detrend.groupby('time.month').mean(dim='time',skipna=True)
mld_anom = mld_detrend.groupby('time.month') - mld_clim
#sst_anom=sst_anom.coarsen(time=3).mean()
mld_anom=wgt_areaave(mld_anom,-90,-50,0,360)
mld_anom=detrend_dim(mld_anom,dim='time')
mld_index=mld_anom/mld_anom.std()

def pdens(S,theta):

    # --- Define constants (Table 1 Column 4, Wright 1997, J. Ocean Tech.)---
    a0 = 7.057924e-4
    a1 = 3.480336e-7
    a2 = -1.112733e-7

    b0 = 5.790749e8
    b1 = 3.516535e6
    b2 = -4.002714e4
    b3 = 2.084372e2
    b4 = 5.944068e5
    b5 = -9.643486e3

    c0 = 1.704853e5
    c1 = 7.904722e2
    c2 = -7.984422
    c3 = 5.140652e-2
    c4 = -2.302158e2
    c5 = -3.079464

    # To compute potential density keep pressure p = 100 kpa
    # S in standard salinity units psu, theta in DegC, p in pascals

    p = 100000.
    alpha0 = a0 + a1*theta + a2*S
    p0 = b0 + b1*theta + b2*theta**2 + b3*theta**3 + b4*S + b5*theta*S
    lambd = c0 + c1*theta + c2*theta**2 + c3*theta**3 + c4*S + c5*theta*S

    pot_dens = (p + p0)/(lambd + alpha0*(p + p0))

    return pot_dens


pt = xr.apply_ufunc(pdens, ds.so, ds.tos,
                    dask='parallelized',
                    output_dtypes=[ds.so.dtype])

rho_ref = 1035.
anom_density = pt - rho_ref

g = 9.81
buoyancy = -g * anom_density / rho_ref

#buoyancy=(buoyancy*area)/ total_area
b_detrend=buoyancy
b_clim = b_detrend.groupby('time.month').mean(dim='time',skipna=True)
b_anom = b_detrend.groupby('time.month') - b_clim
b_anom=wgt_areaave(b_anom,-90,-50,0,360)
b_anom=detrend_dim(b_anom,dim='time')
b_index=b_anom/b_anom.std()


pt=pt#(pt*area)/total_area
pd_detrend=pt
pd_clim = pd_detrend.groupby('time.month').mean(dim='time')
pd_anom = pd_detrend.groupby('time.month') - pd_clim

#zos_anom=zos_anom.coarsen(time=3).mean(
pd_anom=wgt_areaave(pd_anom,-90,-50,0,360)
pd_anom=detrend_dim(pd_anom,dim='time')
pd_index=pd_anom/pd_anom.std()

no3_detrend=ds.no3
no3_clim = no3_detrend.groupby('time.month').mean(dim='time',skipna=True)
no3_anom = no3_detrend.groupby('time.month') - no3_clim
#sst_anom=sst_anom.coarsen(time=3).mean()
no3_anom=wgt_areaave(no3_anom,-90,-50,0,360)
no3_anom=detrend_dim(no3_anom,dim='time')
no3_index=no3_anom/no3_anom.std()

# Example data (replace with your actual data)
# Example data (replace with your actual data)
sst = tos_index#.coarsen(time=2).mean()
salinity = so_index#.coarsen(time=2).mean()
buoyancy = b_index#.coarsen(time=2).mean()

ekman_pumping = ekman_pumping_anom#.coarsen(time=2).mean()
no3 = no3_index#.coarsen(time=2).mean()
pt=pd_index#.coarsen(time=2).mean()
mld=mld_index#.coarsen(time=2).mean()

no3 = no3_index#.coarsen(time=2).mean()

input_series = np.stack((sst, salinity, buoyancy, ekman_pumping, pt,mld), axis=1)
target_series = np.array(no3)


prediction1=np.zeros((4,24))
prediction2=np.zeros((4,24))
prediction3=np.zeros((4,24))
prediction4=np.zeros((4,24))
prediction5=np.zeros((4,24))
prediction6=np.zeros((4,24))
prediction7=np.zeros((4,24))
prediction8=np.zeros((4,24))
prediction9=np.zeros((4,24))
prediction10=np.zeros((4,24))

input2=[48]
output2=[48]
skill_so=np.zeros((4,9))



seed2=list(range(4))

for ii in range(4):
    seed =seed2[ii] #50
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    w=0
    i=0
    Z=[0,12,24,36,48]
    #D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
    D2=np.zeros((444,6))
    for j in range(6):
        D2[:,j]=ts_diff(input_series[:,j])
    #T2[:,]=target_series[:,]
    T2=np.zeros((444,))
    for j in range(1):
        T2[:,]=ts_diff(target_series[:,])
    # Use the mask to exclude NaN value
    
    #D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
    #T2=target_series#np.concatenate((no3_index[:,np.newaxis],tos_index[:,np.newaxis]),axis=1)
#   D=np.concatenate((X_no3[:-Z[i],np.newaxis],X_po4[:-Z[i],np.newaxis],pc1_tos[:-Z[i],:10],pc1_so[:-Z[i],:5],pc1_zos[:-Z[i],:5]),axis=1) 
    X_ss, Y_mm =  split_sequences(D2,T2[:,], input2[0],output2[0])
    print("X_ss",X_ss.shape)
    print("y_mm",Y_mm.shape)
    train_ratio=0.7
    train_len = round(len(X_ss[:-(input2[0]+output2[0]+24)]) * train_ratio)
    test_len=input2[0]+output2[0] #150/3
    X_train, Y_train= X_ss[:-(input2[0]+output2[0]+24)],\
                                   Y_mm[:-(input2[0]+output2[0]+24)],\
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]

    print("X_train",X_train.shape)
    X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]
    x_train = torch.tensor(data = X_train).float()
    y_train = torch.tensor(data = Y_train).float()

    x_val = torch.tensor(data = X_val).float()
    y_val = torch.tensor(data = Y_val).float()

    #x_test = torch.tensor(data = X_test).float()
    #y_test = torch.tensor(data = Y_test).float()
    x_train = x_train.transpose(1, 2)
    x_val = x_val.transpose(1, 2)
    #x_test = x_test.transpose(1, 2)

    #y_train = y_train[:, :, 0]
    #y_val = y_val[:,:,0]
    print("x_train",x_train.shape)
    print("y_train",y_train.shape)
    train_len = x_train.size()[0]

    model_params = {
    'input_size': D2.shape[1], #60
    'output_size':  48,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout,
    'l1_factor':l1_factor,
    'l2_factor':l2_factor,
    'num_heads':3
    }
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNN(**model_params)

    best_params = None
    min_val_loss = sys.maxsize

    training_loss = []
    validation_loss = []

    #model = model.to(device)
    n_splits = 3  # Number of fold

    # Data preparation
    X_ss, Y_mm = split_sequences(D2,target_series[:,], input2[0], output2[0])
    print("X_ss", X_ss.shape)
    print("y_mm", Y_mm.shape)

    # Results storage
    all_train_loss = []
    all_val_loss = []
    #early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Train the model on the current fold
    kernel_sizes = [3,4,5,6]
    channel_sizes_options = [[6,6,6],[9,9,9],[12,12,12],[18,18,18]]
    dropout_list=[0,0.1,0.2,0.3]
    epochs = 200

    best_config = None
    best_val_loss = np.inf
    best_model_params = None

    train_losses=[]
    val_losses=[]
    best_model_path = 'best_model.pth'
    
    for kernel_size in kernel_sizes:
      for channel_sizes in channel_sizes_options:
         for dropout in dropout_list:

            print(f"Testing Kernel Size: {kernel_size}, Channel Sizes: {channel_sizes}")

            model_params = {
            'input_size': 6,  # Adjust this as needed
            'output_size': 48,
            'num_channels': channel_sizes,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'l1_factor': l1_factor,
            'l2_factor': l2_factor,
            'num_heads': 3
            }

            model = TCNN(**model_params)
            optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay=0.000001, lr=0.007) 
            mse_loss = nn.MSELoss()

            early_stopping = EarlyStopping(patience=10, verbose=True)

            # Train the model and get the validation loss
            best_params, train_loss, val_loss, val_loss_min = train_model(
            model, x_train, y_train, epochs, optimizer, mse_loss,early_stopping
            )

            if val_loss_min < best_val_loss:
               best_val_loss = val_loss_min
               best_config = (kernel_size, channel_sizes, dropout)
               best_model_params = best_params

            train_losses.append(train_loss)
            val_losses.append(val_loss)




    # Optionally load the best model parameters
    model.load_state_dict(best_params)
    
    avg_train_loss = np.mean([np.min(losses) for losses in train_loss])
    avg_val_loss = np.mean([np.min(losses) for losses in val_loss])
    plt.figure()
    plt.title('Training Progress')
    plt.yscale("log")
    plt.plot(avg_train_loss, label = 'train')
    plt.plot(avg_val_loss, label = 'validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    plt.savefig("loss_"+str(w)+".png")


    def ts_int(ts_diff, ts_base, start=0):
    #    """
    #    Integrate a differenced time series using cumulative sum.

    #    Parameters:
    #    - ts_diff (numpy array): The differenced time series.
    #    - ts_base (numpy array): The base time series.
    #    - start (float): The initial value for integration.

    #    Returns:
    #    - ts_integrated (numpy array): The integrated time series.
    #    """
        ts_diff = np.asarray(ts_diff)
        ts_base = np.asarray(ts_base)

        ts_integrated = np.empty_like(ts_diff)
        ts_integrated[0] = start + ts_diff[0]

        # Use cumulative sum for integration
        ts_integrated[1:] = np.cumsum(ts_diff[1:]) + ts_base[:-1]
        return ts_integrated.tolist()
    
    #def ts_int(ts_diff, ts_base, start):
#        ts_int = [start]
#        for i in range(1, len(ts_diff)):
#            ts_int.append(ts_int[i-1] + ts_diff[i-1] + ts_base[i-1])
#        return np.array(ts_int)

    for N in range(24):
        if N==0:
           test_len=input2[0]+output2[0]
           X_test, Y_test= X_ss[-48:],Y_mm[-48:]
           x_test = torch.tensor(data = X_test).float()
           y_test = torch.tensor(data = Y_test).float()
           x_test = x_test.transpose(1, 2)
           #y_test=y_test[:,:,ii]

           best_model = TCNN(
           input_size=6,
           output_size=48,
           num_channels=best_config[1],
           kernel_size=best_config[0],
           dropout=best_config[2],
           l1_factor=l1_factor,
           l2_factor=l2_factor,
           num_heads=3
           )
           best_model.eval()
           #best_model.load_state_dict(best_model_params)

           tcn_prediction = best_model(x_test)

           #print('tcn_prediction',tcn_prediction[-1,:].detach().numpy().shape)
           A=0
           years=np.arange(1996-int(A/12),2015,1/12)
           test_len=input2[0]+output2[0]  #150/3

           Z=ts_int(
            tcn_prediction[-1,:].tolist(),
            target_series[-48:,],
            start =  target_series[-48-1,]
            )
           #Q=["NO3 anomaly","PO4 anomaly","first pc of SST"]
           #ci = 0.1 * np.std(Z[input2[0]:]) / np.mean(Z[input2[0]:])
           #95% confidence interval
           #plt.figure()
           #plt.fill_between(years[-108:], (Z[input2[0]:]-ci), (Z[input2[0]:]+ci), color='green', alpha=0.5)
           #plt.plot(years[-108:],Z[input2[0]:],label = 'tcn',color='k',linewidth=2.5)
           #plt.plot(years[-108:],T2[-108:], label = 'real',color='r',linewidth=2.5)
           #plt.ylabel(Q[ii],fontsize=13)
           #plt.xlabel("Years",fontsize=13)
           #plt.xticks(fontsize=13)
           #plt.yticks(fontsize=13)
           #plt.legend()
           #plt.show()
           #plt.savefig('forecast_TCNN_GFDL_0.png')

           
           prediction1[ii,N]=Z[-1]
           prediction2[ii,N]=Z[-6]
           prediction3[ii,N]=Z[-12]
           prediction4[ii,N]=Z[-18]
           prediction5[ii,N]=Z[-24]
           prediction6[ii,N]=Z[-30]
           prediction7[ii,N]=Z[-36]
           prediction8[ii,N]=Z[-42] 
           


        if N>0:
           X_test, Y_test= X_ss[-48-N:-N],Y_mm[-48-N:-N]
           x_test = torch.tensor(data = X_test).float()
           y_test = torch.tensor(data = Y_test).float()
           x_test = x_test.transpose(1, 2)
           
           best_model = TCNN(
           input_size=6,
           output_size=48,
           num_channels=best_config[1],
           kernel_size=best_config[0],
           dropout=best_config[2],
           l1_factor=l1_factor,
           l2_factor=l2_factor,
           num_heads=3
           )
           best_model.eval()
           #best_model.load_state_dict(best_model_params)

           tcn_prediction = best_model(x_test)


           A=24
           test_len=input2[0]+output2[0]  #150/3
           
           Z=ts_int(
            tcn_prediction[-1,:].tolist(),
            target_series[-48-N:-N,],
            start = target_series[-48-N-1,]
            )
           Q=["NO3 anomaly","PO4 anomaly","first pc of SST"]
           ci = 0.1 * np.std(Z[input2[0]:]) / np.mean(Z[input2[0]:])
           #95% confidence interval
           #plt.figure()
           #plt.fill_between(years[-96-N:-N],(Z[20:]-ci), (Z[20:]+ci), color='green', alpha=0.5)
           #plt.plot(years[-108-N:-N],Z[input2[0]:],label = 'tcn',color='k',linewidth=2.5)
           #plt.plot(years[-108-N:-N],T2[-108-N:-N,], label = 'real',color='r',linewidth=2.5)
           #plt.ylabel(Q[ii],fontsize=13)
           #plt.xlabel("Years",fontsize=13)
           #plt.xticks(fontsize=13)
           #plt.yticks(fontsize=13)
           #plt.legend()
           #plt.show( 
              
           prediction1[ii,N]=Z[-1]
           prediction2[ii,N]=Z[-6]
           prediction3[ii,N]=Z[-12]
           prediction4[ii,N]=Z[-18]
           prediction5[ii,N]=Z[-24]
           prediction6[ii,N]=Z[-30]
           prediction7[ii,N]=Z[-36]
           prediction8[ii,N]=Z[-42] 

    Q=6
    A=np.corrcoef(prediction1[ii,::-1],target_series[-24:,])
    skill_so[ii,8]=A[1][0]
    A=np.corrcoef(prediction2[ii,::-1],target_series[-24-Q:-6,])
    skill_so[ii,7]=A[1][0]
    A=np.corrcoef(prediction3[ii,::-1],target_series[-24-Q*2:-12,])
    skill_so[ii,6]=A[1][0]
    A=np.corrcoef(prediction4[ii,::-1],target_series[-24-Q*3:-18,])
    skill_so[ii,5]=A[1][0]
    A=np.corrcoef(prediction5[ii,::-1],target_series[-24-Q*4:-24,])
    skill_so[ii,4]=A[1][0]
    A=np.corrcoef(prediction6[ii,::-1],target_series[-24-Q*5:-30,])
    skill_so[ii,3]=A[1][0]
    A=np.corrcoef(prediction7[ii,::-1],target_series[-24-Q*6:-36,])
    skill_so[ii,2]=A[1][0]
    A=np.corrcoef(prediction8[ii,::-1],target_series[-24-Q*7:-42,])
    skill_so[ii,1]=A[1][0]
    skill_so[ii,0]=1
    
plt.figure(figsize=(10, 10),dpi=1200) 
plt.plot(np.mean(skill_so,axis=0),'r')
plt.ylabel("skill (correlation values)",fontsize=13)
plt.xlabel("Time (years)",fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()

from sklearn.linear_model import LinearRegression
model =LinearRegression()#LinearRegression()
alpha = 0.01  # Regularization strength (adjus


def ts_int(ts_diff, ts_base, start=0):
    #    """
    #    Integrate a differenced time series using cumulative sum.

    #    Parameters:
    #    - ts_diff (numpy array): The differenced time series.
    #    - ts_base (numpy array): The base time series.
    #    - start (float): The initial value for integration.

    #    Returns:
    #    - ts_integrated (numpy array): The integrated time series.
    #    """
        ts_diff = np.asarray(ts_diff)
        ts_base = np.asarray(ts_base)

        ts_integrated = np.empty_like(ts_diff)
        ts_integrated[0] = start + ts_diff[0]

        # Use cumulative sum for integration
        ts_integrated[1:] = np.cumsum(ts_diff[1:]) + ts_base[:-1]
        return ts_integrated.tolist()


# Use the mask to exclude NaN value
input2=[48]
output2=[48]
#D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)

D2=np.zeros((444,6))
for j in range(6):
    D2[:,j]=ts_diff(input_series[:,j])


T2[:,]=ts_diff(target_series[:,])
    
X_ss, Y_mm =  split_sequences(D2,T2, input2[0],output2[0])
print("X_ss",X_ss.shape)
print("y_mm",Y_mm.shape)
train_ratio=0.8
train_len = round(len(X_ss[:-(input2[0]+output2[0]+24)]) * train_ratio)
test_len=input2[0]+output2[0] #150/3
X_train, Y_train= X_ss[:-(input2[0]+output2[0]+24)],\
                                   Y_mm[:-(input2[0]+output2[0]+24)],\
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]


print("X_train",X_train.shape)
X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]

x_train = torch.tensor(data = X_train).float()
y_train = torch.tensor(data = Y_train).float()
x_val = torch.tensor(data = X_val).float()
y_val = torch.tensor(data = Y_val).float()
X_train = X_train.reshape(X_train.shape[0], -1)

X_test,y_test=X_ss[-(48+48+24):],Y_mm[-(48+48+24):]
X_test = X_test.reshape(X_test.shape[0], -1)

model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)




prediction1=np.zeros((24))
prediction2=np.zeros((24))
prediction3=np.zeros((24))
prediction4=np.zeros((24))
prediction5=np.zeros((24))
prediction6=np.zeros((24))
prediction7=np.zeros((24))
prediction8=np.zeros((24))
prediction9=np.zeros((24))

skill_linreg=np.zeros((9))
for N in range(24):
  if N==0:

     X_test, Y_test= X_ss[-48:],Y_mm[-48:]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-48:,],
           start = target_series[-48-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-12]
     prediction3[N]=Z[-12]
     prediction4[N]=Z[-18]
     prediction5[N]=Z[-24]
     prediction6[N]=Z[-30]
     prediction7[N]=Z[-36]
     prediction8[N]=Z[-42] 
     
  if N>0:

     X_test, Y_test= X_ss[-48-N:-N],Y_mm[-48-N:-N]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-48-N:-N,],
           start = target_series[-48-N-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-6]
     prediction3[N]=Z[-12]
     prediction4[N]=Z[-18]
     prediction5[N]=Z[-24]
     prediction6[N]=Z[-30]
     prediction7[N]=Z[-36]
     prediction8[N]=Z[-42] 


Q=6
A=np.corrcoef(prediction1[::-1],target_series[-24:,])
skill_linreg[8]=A[1][0]
A=np.corrcoef(prediction2[::-1],target_series[-24-Q:-6,])
skill_linreg[7]=A[1][0]
A=np.corrcoef(prediction3[::-1],target_series[-24-Q*2:-12,])
skill_linreg[6]=A[1][0]
A=np.corrcoef(prediction4[::-1],target_series[-24-Q*3:-18,])
skill_linreg[5]=A[1][0]
A=np.corrcoef(prediction5[::-1],target_series[-24-Q*4:-24,])
skill_linreg[4]=A[1][0]
A=np.corrcoef(prediction6[::-1],target_series[-24-Q*5:-30,])
skill_linreg[3]=A[1][0]
A=np.corrcoef(prediction7[::-1],target_series[-24-Q*6:-36,])
skill_linreg[2]=A[1][0]
A=np.corrcoef(prediction8[::-1],target_series[-24-Q*7:-42,])
skill_linreg[1]=A[1][0]
skill_linreg[0]=1   






from sklearn.linear_model import Ridge
alpha = 0.00005  # Regularization strength (adjust as needed)
model = Ridge(alpha=alpha)

D2=np.zeros((444,6))
for j in range(6):
    D2[:,j]=ts_diff(input_series[:,j])


T2=ts_diff(target_series)
# Use the mask to exclude NaN value
input2=[48]
output2=[48]
#D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
#T2=target_series
X_ss, Y_mm =  split_sequences(D2,T2, input2[0],output2[0])
print("X_ss",X_ss.shape)
print("y_mm",Y_mm.shape)
train_ratio=0.8
train_len = round(len(X_ss[:-(input2[0]+output2[0]+24)]) * train_ratio)
test_len=input2[0]+output2[0] #150/3
X_train, Y_train= X_ss[:-(input2[0]+output2[0]+24)],\
                                   Y_mm[:-(input2[0]+output2[0]+24)],\
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]


print("X_train",X_train.shape)
X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]

x_train = torch.tensor(data = X_train).float()
y_train = torch.tensor(data = Y_train).float()

x_val = torch.tensor(data = X_val).float()
y_val = torch.tensor(data = Y_val).float()
X_train = X_train.reshape(X_train.shape[0], -1)

X_test,y_test=X_ss[-(48+48+24):],Y_mm[-(48+48+24):]
X_test = X_test.reshape(X_test.shape[0], -1)

model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)




prediction1=np.zeros((24))
prediction2=np.zeros((24))
prediction3=np.zeros((24))
prediction4=np.zeros((24))
prediction5=np.zeros((24))
prediction6=np.zeros((24))
prediction7=np.zeros((24))
prediction8=np.zeros((24))
prediction9=np.zeros((24))


skill_ridge=np.zeros((9))
for N in range(24):
  if N==0:

     X_test, Y_test= X_ss[-48:],Y_mm[-48:]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-48:,],
           start = target_series[-48-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-6]
     prediction3[N]=Z[-12]
     prediction4[N]=Z[-18]
     prediction5[N]=Z[-24]
     prediction6[N]=Z[-30]
     prediction7[N]=Z[-36]
     prediction8[N]=Z[-42] 
     
  if N>0:

     X_test, Y_test= X_ss[-48-N:-N],Y_mm[-48-N:-N]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-48-N:-N,],
           start = target_series[-48-N-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-6]
     prediction3[N]=Z[-12]
     prediction4[N]=Z[-18]
     prediction5[N]=Z[-24]
     prediction6[N]=Z[-30]
     prediction7[N]=Z[-36]
     prediction8[N]=Z[-42] 
   


Q=6
A=np.corrcoef(prediction1[::-1],target_series[-24:,])
skill_ridge[8]=A[1][0]
A=np.corrcoef(prediction2[::-1],target_series[-24-Q:-6,])
skill_ridge[7]=A[1][0]
A=np.corrcoef(prediction3[::-1],target_series[-24-Q*2:-12,])
skill_ridge[6]=A[1][0]
A=np.corrcoef(prediction4[::-1],target_series[-24-Q*3:-18,])
skill_ridge[5]=A[1][0]
A=np.corrcoef(prediction5[::-1],target_series[-24-Q*4:-24,])
skill_ridge[4]=A[1][0]
A=np.corrcoef(prediction6[::-1],target_series[-24-Q*5:-30,])
skill_ridge[3]=A[1][0]
A=np.corrcoef(prediction7[::-1],target_series[-24-Q*6:-36,])
skill_ridge[2]=A[1][0]
A=np.corrcoef(prediction8[::-1],target_series[-24-Q*7:-42,])
skill_ridge[1]=A[1][0]
skill_ridge[0]=1 



lags = range(49)
persistence=np.zeros((1,49))
import statsmodels.api as sm
acorr = sm.tsa.acf(target_series, nlags = len(lags)-1)
#auto2[im,:]=acorr
#acorr = sm.tsa.acf(savitzky_golay(A2[:,im],12,3), nlags = len(lags)-1)
persistence[0,:]=acorr



import numpy as np
import matplotlib.pyplot as plt

# Assuming skill_so, skill_linreg, skill_ridge, persistence are defined arrays

# Calculate standard deviation
ci = np.std(skill_so, axis=0)

# Plotting
plt.figure()
plt.fill_between(range(len(np.mean(skill_so, axis=0))),
                 np.mean(skill_so, axis=0) - ci,
                 np.mean(skill_so, axis=0) + ci,
                 color='green', alpha=0.5, label='Skill_so Spread')

plt.plot(np.mean(skill_so, axis=0), 'r', label='Mean skill_so')
plt.plot(skill_linreg, 'g', label='Skill_linreg')
plt.plot(skill_ridge, 'k', label='Skill_ridge')
plt.plot(persistence[0, ::6], 'b', label='Persistence')
plt.ylabel("Skill", fontsize=13)
plt.xlabel("Time (6-month)", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=6)
plt.title("Comparison of Skills", fontsize=14)
plt.grid(True)
plt.show()

#all data available 

import intake

cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(cat_url)
col
cat = col.search(experiment_id=['historical'], variable_id=['tos','so','no3','thetao','mlotst'],source_id=['GFDL-ESM4'],table_id=['Omon'], grid_label=['gr'])
dset_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
dset_dict.keys()

ds = dset_dict['CMIP.NOAA-GFDL.GFDL-ESM4.historical.Omon.gr']

min_lon = 0
min_lat = -90
min_depth = 0

max_lon = 360
max_lat = -50
max_depth = 50

mask_lat = (ds.lat >= min_lat) & (ds.lat <= max_lat)
mask_depth = (ds.lev >= min_depth) & (ds.lev <= max_depth)
ds = ds.where(mask_lat & mask_depth, drop=True)
ds=ds.mean('lev').mean('dcpp_init_year').mean('member_id')


ekman_pumping, Udy, Vdx = z_curl_xr(taux3 / (rho0 * fx), tauy3 / (rho0 * fy), dx, dy, lat_wsc)
ekman_pumping=ekman_pumping.sel(lat=slice(-90,-50))
ekman_pumping=ekman_pumping.transpose('time','lat','lon')
ekman_pumping['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
#ekman_pumping=ekman_pumping.isel(time=slice(1980-444,1980))

ekman_pumping_clim = ekman_pumping.groupby('time.month').mean(dim='time',skipna=True)
ekman_pumping_anom = ekman_pumping.groupby('time.month') - ekman_pumping_clim 

ekman_pumping_anom = wgt_areaave(ekman_pumping_anom, -90, -50, 0, 360)
ekman_pumping_anom=detrend_dim(ekman_pumping_anom,dim='time')
ekman_pumping_anom

ds['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
#ds=ds.isel(time=slice(1980-444,1980))

so_detrend=ds.so
so_clim = so_detrend.groupby('time.month').mean(dim='time',skipna=True)
so_anom = so_detrend.groupby('time.month') - so_clim
#sst_anom=sst_anom.coarsen(time=3).mean()
so_anom=wgt_areaave(so_anom,-90,-50,0,360)
so_anom=detrend_dim(so_anom,dim='time')
so_index=so_anom/so_anom.std()

tos_detrend=ds.tos
tos_clim = tos_detrend.groupby('time.month').mean(dim='time',skipna=True)
tos_anom = tos_detrend.groupby('time.month') - tos_clim
#sst_anom=sst_anom.coarsen(time=3).mean()
tos_anom=wgt_areaave(tos_anom,-90,-50,0,360)
tos_anom=detrend_dim(tos_anom,dim='time')
tos_index=tos_anom/tos_anom.std()

mld_detrend=ds.mlotst
mld_clim = mld_detrend.groupby('time.month').mean(dim='time',skipna=True)
mld_anom = mld_detrend.groupby('time.month') - mld_clim
#sst_anom=sst_anom.coarsen(time=3).mean()
mld_anom=wgt_areaave(mld_anom,-90,-50,0,360)
mld_anom=detrend_dim(mld_anom,dim='time')
mld_index=mld_anom/mld_anom.std()

def pdens(S,theta):

    # --- Define constants (Table 1 Column 4, Wright 1997, J. Ocean Tech.)---
    a0 = 7.057924e-4
    a1 = 3.480336e-7
    a2 = -1.112733e-7

    b0 = 5.790749e8
    b1 = 3.516535e6
    b2 = -4.002714e4
    b3 = 2.084372e2
    b4 = 5.944068e5
    b5 = -9.643486e3

    c0 = 1.704853e5
    c1 = 7.904722e2
    c2 = -7.984422
    c3 = 5.140652e-2
    c4 = -2.302158e2
    c5 = -3.079464

    # To compute potential density keep pressure p = 100 kpa
    # S in standard salinity units psu, theta in DegC, p in pascals

    p = 100000.
    alpha0 = a0 + a1*theta + a2*S
    p0 = b0 + b1*theta + b2*theta**2 + b3*theta**3 + b4*S + b5*theta*S
    lambd = c0 + c1*theta + c2*theta**2 + c3*theta**3 + c4*S + c5*theta*S

    pot_dens = (p + p0)/(lambd + alpha0*(p + p0))

    return pot_dens


pt = xr.apply_ufunc(pdens, ds.so, ds.tos,
                    dask='parallelized',
                    output_dtypes=[ds.so.dtype])

rho_ref = 1035.
anom_density = pt - rho_ref

g = 9.81
buoyancy = -g * anom_density / rho_ref

#buoyancy=(buoyancy*area)/ total_area
b_detrend=buoyancy
b_clim = b_detrend.groupby('time.month').mean(dim='time',skipna=True)
b_anom = b_detrend.groupby('time.month') - b_clim
b_anom=wgt_areaave(b_anom,-90,-50,0,360)
b_anom=detrend_dim(b_anom,dim='time')
b_index=b_anom/b_anom.std()


pt=pt#(pt*area)/total_area
pd_detrend=pt
pd_clim = pd_detrend.groupby('time.month').mean(dim='time')
pd_anom = pd_detrend.groupby('time.month') - pd_clim

#zos_anom=zos_anom.coarsen(time=3).mean(
pd_anom=wgt_areaave(pd_anom,-90,-50,0,360)
pd_anom=detrend_dim(pd_anom,dim='time')
pd_index=pd_anom/pd_anom.std()

no3_detrend=ds.no3
no3_clim = no3_detrend.groupby('time.month').mean(dim='time',skipna=True)
no3_anom = no3_detrend.groupby('time.month') - no3_clim
#sst_anom=sst_anom.coarsen(time=3).mean()
no3_anom=wgt_areaave(no3_anom,-90,-50,0,360)
no3_anom=detrend_dim(no3_anom,dim='time')
no3_index=no3_anom/no3_anom.std()

# Example data (replace with your actual data)
# Example data (replace with your actual data)
sst = tos_index#.coarsen(time=2).mean()
salinity = so_index#.coarsen(time=2).mean()
buoyancy = b_index#.coarsen(time=2).mean()

ekman_pumping = ekman_pumping_anom#.coarsen(time=2).mean()
no3 = no3_index#.coarsen(time=2).mean()
pt=pd_index#.coarsen(time=2).mean()
mld=mld_index#.coarsen(time=2).mean()

no3 = no3_index#.coarsen(time=2).mean()

input_series = np.stack((sst, salinity, buoyancy, ekman_pumping, pt,mld), axis=1)
target_series = np.array(no3)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
import random
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import copy
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

seed =50 #2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)



# time series input
features =20
# training epochs
epochs =200 #1000
# synthetic time series dataset
ts_len = 1980
# test dataset size
test_len =106
# temporal casual layer channels
channel_sizes = [9] * 3
# convolution kernel size
kernel_size =3 #5
dropout = 0.1


#ts = generate_time_series(ts_len)
train_ratio=0.7
#ts_diff_y = ts_diff(ts[:, 0])
#ts_diff = copy.deepcopy(ts)
#ts_diff[:, 0] = ts_diff_y
l1_factor=3*10^(-10)
l2_factor=3*10^(-10)

C=[0,1,2]
lr2=[0.0001,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001,0.0002,0.0003]



#E=np.concatenate((X_no3[:,np.newaxis],X_po4[:,np.newaxis]),axis=1)
#E=[X_no3[:,np.newaxis],X_po4[:,np.newaxis]]
#print("E",E.shape)



N=24

prediction1=np.zeros((4,264))
prediction2=np.zeros((4,264))
prediction3=np.zeros((4,264))
prediction4=np.zeros((4,264))
prediction5=np.zeros((4,264))
prediction6=np.zeros((4,264))
prediction7=np.zeros((4,264))
prediction8=np.zeros((4,264))
prediction9=np.zeros((4,264))
prediction10=np.zeros((4,264))



#print("X_no3",X_no3.shape)
#print("X_npp",X_npp.shape)


#npp_index=npp_index.T
#print("npp_index_GFDL",npp_index.shape)



k=0
#M=[3,6,9,12,15,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168,174,180,186,192,198,204,210,216,222,228,234,240,246,252,258,264]

#M=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180,183,186,189,192,195,198,201,204,207,210,213,216,219,222,225,228,231,234,237,240,243,246,249,252,255,258,261,264]
M=np.arange(1,24)

years=np.arange(1850,2015,1/12)

class EarlyStopping:

    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt',trace_func=print):

        #Args:
        """
                            Default: False
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.val_loss_min = np.Inf
        self.trace_func = trace_func

        
    def __call__(self, val_loss, model):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        energy = torch.tanh(self.W(x))  # shape: (batch_size, seq_length, hidden_size)
        energy = energy.view(x.size(0), -1, self.num_heads, self.head_size)  # shape: (batch_size, seq_length, num_heads, head_size)
        energy = energy.permute(0, 2, 1, 3)  # shape: (batch_size, num_heads, seq_length, head_size)

        attention_weights = torch.softmax(self.V(energy), dim=2)  # shape: (batch_size, num_heads, seq_length, 1)
        context_vector = torch.sum(attention_weights * energy, dim=2)  # shape: (batch_size, num_heads, head_size)

        context_vector = context_vector.view(x.size(0), -1)  # shape: (batch_size, hidden_size)
        return context_vector


class TemporalCasualLayer(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout ):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride':      1,
            'padding':     padding,
            'dilation':    dilation
        }

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.conv3 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop3 = Crop(padding)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.conv4 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop4 = Crop(padding)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.conv5 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop5 = Crop(padding)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2,
                                 self.conv3, self.crop3, self.relu3, self.dropout3,
                                 self.conv4, self.crop4, self.relu4, self.dropout4
                                 
                                
                                 )
        self.residual = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.residual is not None:
            self.residual.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        y = self.net(x)

        output = self.relu(y + residual)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        energy = torch.tanh(self.W(x))  # shape: (batch_size, seq_length, hidden_size)
        energy = energy.view(x.size(0), -1, self.num_heads, self.head_size)  # shape: (batch_size, seq_length, num_heads, head_size)
        energy = energy.permute(0, 2, 1, 3)  # shape: (batch_size, num_heads, seq_length, head_size)

        attention_weights = torch.softmax(self.V(energy), dim=2)  # shape: (batch_size, num_heads, seq_length, 1)
        context_vector = torch.sum(attention_weights * energy, dim=2)  # shape: (batch_size, num_heads, head_size)

        context_vector = context_vector.view(x.size(0), -1)  # shape: (batch_size, hidden_size)
        return context_vector

class Crop(nn.Module):

    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()
    

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, in_channels, T) where N is the batch_size, and T is the sequence length
        mask = np.array([[1 if i>j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        if input.is_cuda:
            mask = torch.ByteTensor(mask).cuda(input.get_device())
        else:
            mask = torch.ByteTensor(mask)
        # mask = mask.bool()
        
        input = input.permute(0,2,1) # input: [N, T, inchannels]
        keys = self.linear_keys(input) # keys: (N, T, key_size)
        query = self.linear_query(input) # query: (N, T, key_size)
        values = self.linear_values(input) # values: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))

        weight_temp = F.softmax(temp / self.sqrt_key_size, dim=1) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_vert = F.softmax(temp / self.sqrt_key_size, dim=1) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_hori = F.softmax(temp / self.sqrt_key_size, dim=2) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp = (weight_temp_hori + weight_temp_vert)/2
        value_attentioned = torch.bmm(weight_temp, values).permute(0,2,1) # shape: (N, T, value_size)
       
        return value_attentioned, weight_temp # value_attentioned: [N, in_channels, T], weight_temp: [N, T, T]
    

class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.1):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_params = {
            'kernel_size': kernel_size,
            'stride': 1,
            'dropout': dropout
        }

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            tcl_params['dilation'] = dilation
            tcl = TemporalCasualLayer(in_channels, out_channels, **tcl_params)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(SparseMultiHeadAttention, self).__init__()
        assert input_size % num_heads == 0, "Input size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_size = input_size // num_heads

        self.W_query = nn.Linear(input_size, input_size)
        self.W_key = nn.Linear(input_size, input_size)
        self.W_value = nn.Linear(input_size, input_size)

    def forward(self, x):
        batch_size, seq_length, input_size = x.size()
        
        queries = self.W_query(x)  # shape: (batch_size, seq_length, input_size)
        keys = self.W_key(x)  # shape: (batch_size, seq_length, input_size)
        values = self.W_value(x)  # shape: (batch_size, seq_length, input_size)

        # Reshape queries, keys, and values for multi-head attention
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        # Perform attention mechanism
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_size)  # shape: (batch_size, num_heads, seq_length, seq_length)
        attention_weights = torch.softmax(energy, dim=-1)  # shape: (batch_size, num_heads, seq_length, seq_length)
        context_vector = torch.matmul(attention_weights, values)  # shape: (batch_size, num_heads, seq_length, head_size)

        # Rearrange context_vector to match the original shape
        context_vector = context_vector.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, input_size)

        return context_vector   
    
class TCNN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout,l1_factor,l2_factor,num_heads):
        super(TCNN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.conv1 = TemporalConvolutionNetwork(input_channels, [32, 64, 128], kernel_size, dropout)
        self.attention = SparseMultiHeadAttention(num_channels[-1], num_heads)
        #self.attention = MultiHeadAttention(num_channels[-1], attention_hidden_size, num_heads)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.dropout = nn.Dropout(dropout)
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #x_mag = self.fourier_feature_engineering(x)
        y = self.tcn(x)
        y = self.dropout(y)

        # L1 Regularization
        l1_reg = torch.tensor(0.001)
        if self.l1_factor > 0:
            for param in self.parameters():
                l1_reg += torch.norm(param, p=1)

        # L2 Regularization
        l2_reg = torch.tensor(0.001)
        if self.l2_factor > 0:
            for param in self.parameters():
                l2_reg += torch.norm(param, p=2)

        #attended_features = self.attention(y[:, :, -1])
        y = y.transpose(1, 2)
        attended_features = self.attention(y)
        attended_features = attended_features.transpose(1, 2)  # Restore original shape
        out = self.linear(attended_features[:, :, -1])

        if self.l1_factor > 0:
            out += self.l1_factor * l1_reg

        if self.l2_factor > 0:
            out += 0.5 * self.l2_factor * l2_reg

        return out  
      

from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import copy


def train_model(model, X, y, epochs, optimizer, mse_loss, early_stopping, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params = None
    min_val_loss = np.inf

    all_training_losses = []
    all_validation_losses = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        x_train, x_val = torch.tensor(X_train).float(), torch.tensor(X_val).float()
        y_train, y_val = torch.tensor(y_train).float(), torch.tensor(y_val).float()

        fold_training_losses = []
        fold_validation_losses = []
        fold_min_val_loss = np.inf

        for t in range(epochs):
            model.train()
            optimizer.zero_grad()

            prediction = model(x_train)
            loss = mse_loss(prediction, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_prediction = model(x_val)
                val_loss = mse_loss(val_prediction, y_val)

            fold_training_losses.append(loss.item())
            fold_validation_losses.append(val_loss.item())

            if val_loss.item() < fold_min_val_loss:
                best_fold_params = copy.deepcopy(model.state_dict())
                fold_min_val_loss = val_loss.item()
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {t} in fold {fold}")
                break

            if t % 10 == 0:
                print(f'Fold {fold}, Epoch {t}. Train Loss: {round(loss.item(), 4)}, Val Loss: {round(val_loss.item(), 4)}')

        all_training_losses.append(fold_training_losses)
        all_validation_losses.append(fold_validation_losses)

        if fold_min_val_loss < min_val_loss:
            best_params = best_fold_params
            min_val_loss = fold_min_val_loss

    return best_params, all_training_losses, all_validation_losses, min_val_loss




def ts_diff(ts):
    diff_ts = [0] * len(ts)
    for i in range(1, len(ts)):
        diff_ts[i] = ts[i] - ts[i - 1]
    return diff_ts


input2=[108]
output2=[108]
skill_so=np.zeros((4,10))



seed2=list(range(4))

for ii in range(4):
    seed =seed2[ii] #50
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    w=0
    i=0
    Z=[0,12,24,36,48]
    #D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
    D2=np.zeros((1980,6))
    for j in range(6):
        D2[:,j]=ts_diff(input_series[:,j])
    #T2[:,]=target_series[:,]
    T2=np.zeros((1980,))
    for j in range(1):
        T2[:,]=ts_diff(target_series[:,])
    # Use the mask to exclude NaN value
    
    #D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
    #T2=target_series#np.concatenate((no3_index[:,np.newaxis],tos_index[:,np.newaxis]),axis=1)
#   D=np.concatenate((X_no3[:-Z[i],np.newaxis],X_po4[:-Z[i],np.newaxis],pc1_tos[:-Z[i],:10],pc1_so[:-Z[i],:5],pc1_zos[:-Z[i],:5]),axis=1) 
    X_ss, Y_mm =  split_sequences(D2,T2[:,], input2[0],output2[0])
    print("X_ss",X_ss.shape)
    print("y_mm",Y_mm.shape)
    train_ratio=0.75
    train_len = round(len(X_ss[:-(input2[0]+output2[0]+264)]) * train_ratio)
    test_len=input2[0]+output2[0] #150/3
    X_train, Y_train= X_ss[:-(input2[0]+output2[0]+264)],\
                                   Y_mm[:-(input2[0]+output2[0]+264)],\
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]

    print("X_train",X_train.shape)
    X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]
    x_train = torch.tensor(data = X_train).float()
    y_train = torch.tensor(data = Y_train).float()

    x_val = torch.tensor(data = X_val).float()
    y_val = torch.tensor(data = Y_val).float()

    #x_test = torch.tensor(data = X_test).float()
    #y_test = torch.tensor(data = Y_test).float()
    x_train = x_train.transpose(1, 2)
    x_val = x_val.transpose(1, 2)
    #x_test = x_test.transpose(1, 2)

    #y_train = y_train[:, :, 0]
    #y_val = y_val[:,:,0]
    print("x_train",x_train.shape)
    print("y_train",y_train.shape)
    train_len = x_train.size()[0]

    model_params = {
    'input_size': D2.shape[1], #60
    'output_size':  108,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout,
    'l1_factor':l1_factor,
    'l2_factor':l2_factor,
    'num_heads':3
    }
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNN(**model_params)

    best_params = None
    min_val_loss = sys.maxsize

    training_loss = []
    validation_loss = []

    #model = model.to(device)
    n_splits = 3  # Number of fold

    # Data preparation
    X_ss, Y_mm = split_sequences(D2,target_series[:,], input2[0], output2[0])
    print("X_ss", X_ss.shape)
    print("y_mm", Y_mm.shape)

    # Results storage
    all_train_loss = []
    all_val_loss = []
    #early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Train the model on the current fold
    kernel_sizes = [3,4,5,6]
    channel_sizes_options = [[6,6,6],[9,9,9],[12,12,12],[18,18,18]]
    dropout_list=[0,0.1,0.2,0.3]
    epochs = 200

    best_config = None
    best_val_loss = np.inf
    best_model_params = None

    train_losses=[]
    val_losses=[]
    best_model_path = 'best_model.pth'
    
    for kernel_size in kernel_sizes:
      for channel_sizes in channel_sizes_options:
         for dropout in dropout_list:

            print(f"Testing Kernel Size: {kernel_size}, Channel Sizes: {channel_sizes}")

            model_params = {
            'input_size': 6,  # Adjust this as needed
            'output_size': 108,
            'num_channels': channel_sizes,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'l1_factor': l1_factor,
            'l2_factor': l2_factor,
            'num_heads': 3
            }

            model = TCNN(**model_params)
            optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay=0.000001, lr=0.007) 
            mse_loss = nn.MSELoss()

            early_stopping = EarlyStopping(patience=10, verbose=True)

            # Train the model and get the validation loss
            best_params, train_loss, val_loss, val_loss_min = train_model(
            model, x_train, y_train, epochs, optimizer, mse_loss,early_stopping
            )

            if val_loss_min < best_val_loss:
               best_val_loss = val_loss_min
               best_config = (kernel_size, channel_sizes, dropout)
               best_model_params = best_params

            train_losses.append(train_loss)
            val_losses.append(val_loss)




    # Optionally load the best model parameters
    model.load_state_dict(best_params)
    
    avg_train_loss = np.mean([np.min(losses) for losses in train_loss])
    avg_val_loss = np.mean([np.min(losses) for losses in val_loss])
    plt.figure()
    plt.title('Training Progress')
    plt.yscale("log")
    plt.plot(avg_train_loss, label = 'train')
    plt.plot(avg_val_loss, label = 'validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    plt.savefig("loss_"+str(w)+".png")


    def ts_int(ts_diff, ts_base, start=0):
    #    """
    #    Integrate a differenced time series using cumulative sum.

    #    Parameters:
    #    - ts_diff (numpy array): The differenced time series.
    #    - ts_base (numpy array): The base time series.
    #    - start (float): The initial value for integration.

    #    Returns:
    #    - ts_integrated (numpy array): The integrated time series.
    #    """
        ts_diff = np.asarray(ts_diff)
        ts_base = np.asarray(ts_base)

        ts_integrated = np.empty_like(ts_diff)
        ts_integrated[0] = start + ts_diff[0]

        # Use cumulative sum for integration
        ts_integrated[1:] = np.cumsum(ts_diff[1:]) + ts_base[:-1]
        return ts_integrated.tolist()
    
    #def ts_int(ts_diff, ts_base, start):
#        ts_int = [start]
#        for i in range(1, len(ts_diff)):
#            ts_int.append(ts_int[i-1] + ts_diff[i-1] + ts_base[i-1])
#        return np.array(ts_int)

    for N in range(264):
        if N==0:
           test_len=input2[0]+output2[0]
           X_test, Y_test= X_ss[-108:],Y_mm[-108:]
           x_test = torch.tensor(data = X_test).float()
           y_test = torch.tensor(data = Y_test).float()
           x_test = x_test.transpose(1, 2)
           #y_test=y_test[:,:,ii]

           best_model = TCNN(
           input_size=6,
           output_size=108,
           num_channels=best_config[1],
           kernel_size=best_config[0],
           dropout=best_config[2],
           l1_factor=l1_factor,
           l2_factor=l2_factor,
           num_heads=3
           )
           best_model.eval()
           #best_model.load_state_dict(best_model_params)

           tcn_prediction = best_model(x_test)

           #print('tcn_prediction',tcn_prediction[-1,:].detach().numpy().shape)
           A=0
           years=np.arange(1996-int(A/12),2015,1/12)
           test_len=input2[0]+output2[0]  #150/3

           Z=ts_int(
            tcn_prediction[-1,:].tolist(),
            target_series[-108:,],
            start =  target_series[-108-1,]
            )
           #Q=["NO3 anomaly","PO4 anomaly","first pc of SST"]
           #ci = 0.1 * np.std(Z[input2[0]:]) / np.mean(Z[input2[0]:])
           #95% confidence interval
           #plt.figure()
           #plt.fill_between(years[-108:], (Z[input2[0]:]-ci), (Z[input2[0]:]+ci), color='green', alpha=0.5)
           #plt.plot(years[-108:],Z[input2[0]:],label = 'tcn',color='k',linewidth=2.5)
           #plt.plot(years[-108:],T2[-108:], label = 'real',color='r',linewidth=2.5)
           #plt.ylabel(Q[ii],fontsize=13)
           #plt.xlabel("Years",fontsize=13)
           #plt.xticks(fontsize=13)
           #plt.yticks(fontsize=13)
           #plt.legend()
           #plt.show()
           #plt.savefig('forecast_TCNN_GFDL_0.png')

           
           prediction1[ii,N]=Z[-1]
           prediction2[ii,N]=Z[-12]
           prediction3[ii,N]=Z[-24]
           prediction4[ii,N]=Z[-36]
           prediction5[ii,N]=Z[-48]
           prediction6[ii,N]=Z[-60]
           prediction7[ii,N]=Z[-72]
           prediction8[ii,N]=Z[-84] 
           prediction9[ii,N]=Z[-96]


        if N>0:
           X_test, Y_test= X_ss[-108-N:-N],Y_mm[-108-N:-N]
           x_test = torch.tensor(data = X_test).float()
           y_test = torch.tensor(data = Y_test).float()
           x_test = x_test.transpose(1, 2)
           
           best_model = TCNN(
           input_size=6,
           output_size=108,
           num_channels=best_config[1],
           kernel_size=best_config[0],
           dropout=best_config[2],
           l1_factor=l1_factor,
           l2_factor=l2_factor,
           num_heads=3
           )
           best_model.eval()
           #best_model.load_state_dict(best_model_params)

           tcn_prediction = best_model(x_test)


           A=24
           test_len=input2[0]+output2[0]  #150/3
           
           Z=ts_int(
            tcn_prediction[-1,:].tolist(),
            target_series[-108-N:-N,],
            start = target_series[-108-N-1,]
            )
           Q=["NO3 anomaly","PO4 anomaly","first pc of SST"]
           ci = 0.1 * np.std(Z[input2[0]:]) / np.mean(Z[input2[0]:])
           #95% confidence interval
           #plt.figure()
           #plt.fill_between(years[-96-N:-N],(Z[20:]-ci), (Z[20:]+ci), color='green', alpha=0.5)
           #plt.plot(years[-108-N:-N],Z[input2[0]:],label = 'tcn',color='k',linewidth=2.5)
           #plt.plot(years[-108-N:-N],T2[-108-N:-N,], label = 'real',color='r',linewidth=2.5)
           #plt.ylabel(Q[ii],fontsize=13)
           #plt.xlabel("Years",fontsize=13)
           #plt.xticks(fontsize=13)
           #plt.yticks(fontsize=13)
           #plt.legend()
           #plt.show( 
              
           prediction1[ii,N]=Z[-1]
           prediction2[ii,N]=Z[-12]
           prediction3[ii,N]=Z[-24]
           prediction4[ii,N]=Z[-36]
           prediction5[ii,N]=Z[-48]
           prediction6[ii,N]=Z[-60]
           prediction7[ii,N]=Z[-72]
           prediction8[ii,N]=Z[-84]
           prediction9[ii,N]=Z[-96]

    Q=12
    A=np.corrcoef(prediction1[ii,::-1],target_series[-264:,])
    skill_so[ii,9]=A[1][0]
    A=np.corrcoef(prediction2[ii,::-1],target_series[-264-Q:-12,])
    skill_so[ii,8]=A[1][0]
    A=np.corrcoef(prediction3[ii,::-1],target_series[-264-Q*2:-24,])
    skill_so[ii,7]=A[1][0]
    A=np.corrcoef(prediction4[ii,::-1],target_series[-264-Q*3:-36,])
    skill_so[ii,6]=A[1][0]
    A=np.corrcoef(prediction5[ii,::-1],target_series[-264-Q*4:-48,])
    skill_so[ii,5]=A[1][0]
    A=np.corrcoef(prediction6[ii,::-1],target_series[-264-Q*5:-60,])
    skill_so[ii,4]=A[1][0]
    A=np.corrcoef(prediction7[ii,::-1],target_series[-264-Q*6:-72,])
    skill_so[ii,3]=A[1][0]
    A=np.corrcoef(prediction8[ii,::-1],target_series[-264-Q*7:-84,])
    skill_so[ii,2]=A[1][0]
    A=np.corrcoef(prediction9[ii,::-1],target_series[-264-Q*8:-96,])
    skill_so[ii,1]=A[1][0]
    skill_so[ii,0]=1
    
plt.figure(figsize=(10, 10),dpi=1200) 
plt.plot(np.mean(skill_so,axis=0),'r')
plt.ylabel("skill (correlation values)",fontsize=13)
plt.xlabel("Time (years)",fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()
    
years=np.arange(1850,2015,1/12)

prediction8[ii,::-1],target_series[-264-Q*7:-84,] # 2
prediction6[ii,::-1],target_series[-264-Q*5:-60,] # 4
prediction1[ii,::-1],target_series[-264:,] # 9

plt.figure()
for i in range(3):
    #plt.fill_between(years[-264:],(prediction1[::-1]-ci), (prediction1[::-1]+ci), color='green', alpha=0.5)
    plt.plot(years[-264:],prediction1[i,::-1],label = 'tcn',color='g',linewidth=0.5)
    plt.plot(years[-264:],np.mean(prediction1[:,::-1],axis=0),label = 'tcn',color='k',linewidth=1.75)
    plt.plot(years[-264:],target_series[-264:,], label = 'real',color='r',linewidth=1.75)
    plt.ylabel("NO3 anomaly",fontsize=13)
    plt.xlabel("Years",fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)


plt.figure()
#plt.fill_between(years[-264-Q*4:-Q*4],(prediction5[::-1]-ci), (prediction5[::-1]+ci), color='green', alpha=0.5)
for i in range(3):
    plt.plot(years[-264-Q*5:-Q*5],prediction6[i,::-1],label = 'tcn',color='g',linewidth=0.5)
    plt.plot(years[-264-Q*5:-Q*5],np.mean(prediction6[:,::-1],axis=0),label = 'tcn',color='k',linewidth=1.75)
    plt.plot(years[-264-Q*5:-Q*5],target_series[-264-Q*5:-60,], label = 'real',color='r',linewidth=1.75)
    plt.ylabel("NO3 anomaly",fontsize=13)
    plt.xlabel("Years",fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
plt.figure()
#plt.fill_between(years[-264-Q*4:-Q*4],(prediction5[::-1]-ci), (prediction5[::-1]+ci), color='green', alpha=0.5)
for i in range(3):
    plt.plot(years[-264-Q*7:-Q*7],prediction8[i,::-1],label = 'tcn',color='g',linewidth=0.5)
    plt.plot(years[-264-Q*7:-Q*7],np.mean(prediction8[:,::-1],axis=0),label = 'tcn',color='k',linewidth=1.75)
    plt.plot(years[-264-Q*7:-Q*7],target_series[-264-Q*7:-Q*7,], label = 'real',color='r',linewidth=1.75)
    plt.ylabel("NO3 anomaly",fontsize=13)
    plt.xlabel("Years",fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)


from sklearn.linear_model import LinearRegression
model =LinearRegression()#LinearRegression()
alpha = 0.01  # Regularization strength (adjus


def ts_int(ts_diff, ts_base, start=0):
    #    """
    #    Integrate a differenced time series using cumulative sum.

    #    Parameters:
    #    - ts_diff (numpy array): The differenced time series.
    #    - ts_base (numpy array): The base time series.
    #    - start (float): The initial value for integration.

    #    Returns:
    #    - ts_integrated (numpy array): The integrated time series.
    #    """
        ts_diff = np.asarray(ts_diff)
        ts_base = np.asarray(ts_base)

        ts_integrated = np.empty_like(ts_diff)
        ts_integrated[0] = start + ts_diff[0]

        # Use cumulative sum for integration
        ts_integrated[1:] = np.cumsum(ts_diff[1:]) + ts_base[:-1]
        return ts_integrated.tolist()


# Use the mask to exclude NaN value
input2=[108]
output2=[108]
#D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)

D2=np.zeros((1980,6))
for j in range(6):
    D2[:,j]=ts_diff(input_series[:,j])

T2=np.zeros((1980,))
T2[:,]=ts_diff(target_series)
    
X_ss, Y_mm =  split_sequences(D2,T2, input2[0],output2[0])
print("X_ss",X_ss.shape)
print("y_mm",Y_mm.shape)
train_ratio=0.7
train_len = round(len(X_ss[:-(input2[0]+output2[0]+264)]) * train_ratio)
test_len=input2[0]+output2[0] #150/3
X_train, Y_train= X_ss[:-(input2[0]+output2[0]+264)],\
                                   Y_mm[:-(input2[0]+output2[0]+264)],\
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]


print("X_train",X_train.shape)
X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]

x_train = torch.tensor(data = X_train).float()
y_train = torch.tensor(data = Y_train).float()
x_val = torch.tensor(data = X_val).float()
y_val = torch.tensor(data = Y_val).float()
X_train = X_train.reshape(X_train.shape[0], -1)

X_test,y_test=X_ss[-(108+108+264):],Y_mm[-(108+108+264):]
X_test = X_test.reshape(X_test.shape[0], -1)

model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)




prediction1=np.zeros((264))
prediction2=np.zeros((264))
prediction3=np.zeros((264))
prediction4=np.zeros((264))
prediction5=np.zeros((264))
prediction6=np.zeros((264))
prediction7=np.zeros((264))
prediction8=np.zeros((264))
prediction9=np.zeros((264))



skill_linreg=np.zeros((10))
for N in range(264):
  if N==0:

     X_test, Y_test= X_ss[-108:],Y_mm[-108:]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-108:,],
           start = target_series[-108-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-12]
     prediction3[N]=Z[-24]
     prediction4[N]=Z[-36]
     prediction5[N]=Z[-48]
     prediction6[N]=Z[-60]
     prediction7[N]=Z[-72]
     prediction8[N]=Z[-84]
     prediction9[N]=Z[-96]
     
  if N>0:

     X_test, Y_test= X_ss[-108-N:-N],Y_mm[-108-N:-N]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-108-N:-N,],
           start = target_series[-108-N-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-12]
     prediction3[N]=Z[-24]
     prediction4[N]=Z[-36]
     prediction5[N]=Z[-48]
     prediction6[N]=Z[-60]
     prediction7[N]=Z[-72]
     prediction8[N]=Z[-84] 
     prediction9[N]=Z[-96]


Q=12
A=np.corrcoef(prediction1[::-1],target_series[-264:,])
skill_linreg[9]=A[1][0]
A=np.corrcoef(prediction2[::-1],target_series[-264-Q:-12,])
skill_linreg[8]=A[1][0]
A=np.corrcoef(prediction3[::-1],target_series[-264-Q*2:-24,])
skill_linreg[7]=A[1][0]
A=np.corrcoef(prediction4[::-1],target_series[-264-Q*3:-36,])
skill_linreg[6]=A[1][0]
A=np.corrcoef(prediction5[::-1],target_series[-264-Q*4:-48,])
skill_linreg[5]=A[1][0]
A=np.corrcoef(prediction6[::-1],target_series[-264-Q*5:-60,])
skill_linreg[4]=A[1][0]
A=np.corrcoef(prediction7[::-1],target_series[-264-Q*6:-72,])
skill_linreg[3]=A[1][0]
A=np.corrcoef(prediction8[::-1],target_series[-264-Q*7:-84,])
skill_linreg[2]=A[1][0]
A=np.corrcoef(prediction9[::-1],target_series[-264-Q*8:-96,])
skill_linreg[1]=A[1][0]
skill_linreg[0]=1   






from sklearn.linear_model import Ridge
alpha = 0.000000000000005  # Regularization strength (adjust as needed)
model = Ridge(alpha=alpha)

D2=np.zeros((1980,6))
for j in range(6):
    D2[:,j]=ts_diff(input_series[:,j])
T2=np.zeros((1980,))

T2=ts_diff(target_series)
# Use the mask to exclude NaN value
input2=[108]
output2=[108]
#D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
#T2=target_series
X_ss, Y_mm =  split_sequences(D2,T2, input2[0],output2[0])
print("X_ss",X_ss.shape)
print("y_mm",Y_mm.shape)
train_ratio=0.7
train_len = round(len(X_ss[:-(input2[0]+output2[0]+264)]) * train_ratio)
test_len=input2[0]+output2[0] #150/3
X_train, Y_train= X_ss[:-(input2[0]+output2[0]+264)],\
                                   Y_mm[:-(input2[0]+output2[0]+264)],\
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]


print("X_train",X_train.shape)
X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]

x_train = torch.tensor(data = X_train).float()
y_train = torch.tensor(data = Y_train).float()

x_val = torch.tensor(data = X_val).float()
y_val = torch.tensor(data = Y_val).float()
X_train = X_train.reshape(X_train.shape[0], -1)

X_test,y_test=X_ss[-(108+108+264):],Y_mm[-(108+108+264):]
X_test = X_test.reshape(X_test.shape[0], -1)

model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)




prediction1=np.zeros((264))
prediction2=np.zeros((264))
prediction3=np.zeros((264))
prediction4=np.zeros((264))
prediction5=np.zeros((264))
prediction6=np.zeros((264))
prediction7=np.zeros((264))
prediction8=np.zeros((264))
prediction9=np.zeros((264))


skill_ridge=np.zeros((10))
for N in range(264):
  if N==0:

     X_test, Y_test= X_ss[-108:],Y_mm[-108:]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-108:,],
           start = target_series[-108-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-12]
     prediction3[N]=Z[-24]
     prediction4[N]=Z[-36]
     prediction5[N]=Z[-48]
     prediction6[N]=Z[-60]
     prediction7[N]=Z[-72]
     prediction8[N]=Z[-84] 
     prediction9[N]=Z[-96]
     
  if N>0:

     X_test, Y_test= X_ss[-108-N:-N],Y_mm[-108-N:-N]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           target_series[-108-N:-N,],
           start = target_series[-108-N-1,]
           )
     prediction1[N]=Z[-1]
     prediction2[N]=Z[-12]
     prediction3[N]=Z[-24]
     prediction4[N]=Z[-36]
     prediction5[N]=Z[-48]
     prediction6[N]=Z[-60]
     prediction7[N]=Z[-72]
     prediction8[N]=Z[-84] 
     prediction9[N]=Z[-96]


Q=12
A=np.corrcoef(prediction1[::-1],target_series[-264:,])
skill_ridge[9]=A[1][0]
A=np.corrcoef(prediction2[::-1],target_series[-264-Q:-12,])
skill_ridge[8]=A[1][0]
A=np.corrcoef(prediction3[::-1],target_series[-264-Q*2:-24,])
skill_ridge[7]=A[1][0]
A=np.corrcoef(prediction4[::-1],target_series[-264-Q*3:-36,])
skill_ridge[6]=A[1][0]
A=np.corrcoef(prediction5[::-1],target_series[-264-Q*4:-48,])
skill_ridge[5]=A[1][0]
A=np.corrcoef(prediction6[::-1],target_series[-264-Q*5:-60,])
skill_ridge[4]=A[1][0]
A=np.corrcoef(prediction7[::-1],target_series[-264-Q*6:-72,])
skill_ridge[3]=A[1][0]
A=np.corrcoef(prediction8[::-1],target_series[-264-Q*7:-84,])
skill_ridge[2]=A[1][0]
A=np.corrcoef(prediction9[::-1],target_series[-264-Q*8:-96,])
skill_ridge[1]=A[1][0]
skill_ridge[0]=1   




lags = range(109)
persistence=np.zeros((1,109))
import statsmodels.api as sm
acorr = sm.tsa.acf(target_series, nlags = len(lags)-1)
#auto2[im,:]=acorr
#acorr = sm.tsa.acf(savitzky_golay(A2[:,im],12,3), nlags = len(lags)-1)
persistence[0,:]=acorr



import numpy as np
import matplotlib.pyplot as plt

# Assuming skill_so, skill_linreg, skill_ridge, persistence are defined arrays

# Calculate standard deviation
ci = np.std(skill_so, axis=0)

# Plotting
plt.figure()
plt.fill_between(range(len(np.mean(skill_so, axis=0))),
                 np.mean(skill_so, axis=0) - ci,
                 np.mean(skill_so, axis=0) + ci,
                 color='green', alpha=0.5, label='Skill_so Spread')

plt.plot(np.mean(skill_so, axis=0), 'r', label='Mean skill_so')
plt.plot(skill_linreg, 'g', label='Skill_linreg')
plt.plot(skill_ridge, 'k', label='Skill_ridge')
plt.plot(persistence[0, ::12], 'b', label='Persistence')
plt.ylabel("Skill", fontsize=13)
plt.xlabel("Time (Years)", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=6)
plt.title("Comparison of Skills", fontsize=14)
plt.grid(True)
plt.show()

