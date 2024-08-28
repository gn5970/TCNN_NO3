import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib import ticker
from cmcrameri import cm
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import intake
import xesmf as xe
import numpy as np
import cartopy
import numpy.polynomial.polynomial as poly
from netCDF4 import Dataset
import shap
import torch
from torch.autograd import grad
import xesmf as xe
from xgcm import Grid
import xesmf as xe
from sklearn.model_selection import train_test_split
from scipy import stats
import cartopy.feature as cfeature
#from geocat.viz import cmaps as gvcmaps
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from scipy.stats import ttest_rel
from scipy.stats import norm

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


def average_da(self, dim=None, weights=None):
    """
    weighted average for DataArrays

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply average.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of self.

    Returns
    -------
    reduced : DataArray
        New DataArray with average applied to its data and the indicated
        dimension(s) removed.

    """

    if weights is None:
        return self.mean(dim)
    else:
        if not isinstance(weights, xray.DataArray):
            raise ValueError("weights must be a DataArray")

        # if NaNs are present, we need individual weights
        if self.notnull().any():
            total_weights = weights.where(self.notnull()).sum(dim=dim)
        else:
            total_weights = weights.sum(dim)

        return (self * weights).sum(dim) / total_weights

def average_ds(self, dim=None, weights=None):
    """
    weighted average for Datasets

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply average.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset
        New Dataset with average applied to its data and the indicated
        dimension(s) removed.

    """

    if weights is None:
        return self.mean(dim)
    else:
        return self.apply(average_da, dim=dim, weights=weights)
    

def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    """

    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid 
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r

def calc_w_from_convergence(u_var, v_var, wrapx = True, wrapy = False):

  tmax = u_var.shape[0]

  ntime, nk, nlat, nlon = u_var.shape
  w = np.ma.zeros( (ntime, nk+1, nlat, nlon)  )
  # Work timelevel by timelevel
  for tidx in range(0,tmax):
    # Get and process the u component
    u_dat = u_var[tidx,:,:,:]
    #h_mask = np.logical_or(np.ma.getmask(u_dat), np.ma.getmask(np.roll(u_dat,1,axis=-1)))
    #u_dat = u_dat.filled(0.)

    # Get and process the v component
    v_dat = v_var[tidx,:,:,:]
    #h_mask = np.logical_or(h_mask,np.ma.getmask(v_dat))
    #h_mask = np.logical_or(h_mask,np.ma.getmask(np.roll(v_dat,1,axis=-2)))
    #v_dat = v_dat.filled(0.)

    # Order of subtraction based on upwind sign convention and desire for w>0 to correspond with upwards velocity
    w[tidx,:-1,:,:] += np.roll(u_dat,1,axis=-1)-u_dat
    if not wrapx: # If not wrapping, then convergence on westernmost side is simply so subtract back the rolled value
      w[tidx,:-1,:,0] += -u_dat[:-1,:,-1]
    w[tidx,:-1,:,:] += np.roll(v_dat,1,axis=-2)-v_dat
    if not wrapy: # If not wrapping, convergence on westernmost side is v
      w[tidx,:-1,0,:] += -v_dat[:,-1,:]
    w[tidx,-1,:,:] = 0.
    # Do a double-flip so that we integrate from the bottom
    w[tidx,:-1,:,:] = w[tidx,-2::-1,:,:].cumsum(axis=0)[::-1,:,:]
    # Mask if any of u[i-1], u[i], v[j-1], v[j] are not masked
    #w[tidx,:-1,:,:] = np.ma.masked_where(h_mask, w[tidx,:-1,:,:])
    # Bottom should always be zero, mask applied wherever the top interface is a valid value
    #w[tidx,-1,:,:] = np.ma.masked_where(h_mask[-2,:,:], w[tidx,-1,:,:])

  return w

def wgt_areaave(indat, latS, latN, lonW, lonE):
  lat=indat.lat
  lon=indat.lon

  if ( ((lonW < 0) or (lonE < 0 )) and (lon.values.min() > -1) ):
     anm=indat.assign_coords(lon=( (lon + 180) % 360 - 180) )
     lon=( (lon + 180) % 360 - 180)
  else:
     anm=indat
  valid_mask = ~np.isnan(anm)
  iplat = lat.where( (lat >= latS ) & (lat <= latN), drop=True)
  iplon = lon.where( (lon >= lonW ) & (lon <= lonE), drop=True)

#  print(iplon)
  wgt = np.cos(np.deg2rad(lat))
  odat=anm.where(valid_mask).sel(lat=iplat,lon=iplon).weighted(wgt).mean(("lon", "lat"), skipna=True)
  return(odat)

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def remove_time_mean(x):
    return x - x.mean(dim='time',skipna=True)

def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([0, 360, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

# New axis constant (actually a reference to *None* behind the scenes)
_NA = np.newaxis

class EofError(Exception):
    """Generic exception class for errors in the eof2 package."""
    pass


class EofToolError(Exception):
    """Generic exception class for errors in the supplementary tools."""
    pass

def _check_flat_center(pcs, field):
    """
    Check PCs and a field for shape compatibility, flatten both to 2D,
    and center along the first dimension.
    This set of operations is common to both covariance and correlation
    calculations.
    """
    # Get the number of times in the field.
    records = field.shape[0]
    if records != pcs.shape[0]:
        # Raise an error if the field has a different number of times to the
        # PCs provided.
        raise EofToolError("PCs and field must have the same first dimension")
    if len(pcs.shape) > 2:
        # Raise an error if the PCs are more than 2D.
        raise EofToolError("PCs must be 1D or 2D")
    # Check if the field is 1D.
    if len(field.shape) == 1:
        field_oned = True
        originalshape = tuple()
        channels = 1
    else:
        # Record the shape of the field and the number of spatial elements.
        originalshape = field.shape[1:]
        channels = np.product(originalshape)
    # Record the number of PCs.
    if len(pcs.shape) == 1:
        npcs = 1
        npcs_out = tuple()
    else:
        npcs = pcs.shape[1]
        npcs_out = (npcs,)
    # Create a flattened field so iterating over space is simple. Also do this
    # for the PCs to ensure they are 2D.
    field_flat = field.reshape([records, channels])
    pcs_flat = pcs.reshape([records, npcs])
    # Centre both the field and PCs in the time dimension.
    field_flat = field_flat - field_flat.mean(axis=0)
    pcs_flat = pcs_flat - pcs_flat.mean(axis=0)
    return pcs_flat, field_flat, npcs_out + originalshape

def correlation_map(pcs, field):
    """Correlation maps for a set of PCs and a spatial-temporal field.
    Given an array where the columns are PCs (e.g., as output from
    :py:meth:`eof2.EofSolve.pcs`) and an array containing a
    spatial-temporal where time is the first dimension, one correlation
    map per PC is computed.
    The field must have the same temporal dimension as the PCs. Any
    number of spatial dimensions (including zero) are allowed in the
    field and there can be any number of PCs.
    **Arguments:**
    *pcs*
        PCs as the columns of an array.
    *field*
        Spatial-temporal field with time as the first dimension.
    """
    # Check PCs and fields for validity, flatten the arrays ready for the
    # computation and remove the mean along the leading dimension.
    pcs_cent, field_cent, out_shape = _check_flat_center(pcs, field)
    # Compute the standard deviation of the PCs and the fields along the time
    # dimension (the leading dimension).
    pcs_std = pcs_cent.std(axis=0)
    field_std = field_cent.std(axis=0)
    # Set the divisor.
    div = np.float64(pcs_cent.shape[0])
    # Compute the correlation map.
    cor = ma.dot(field_cent.T, pcs_cent).T / div
    cor /= ma.outer(pcs_std, field_std)
    # Return the correlation with the appropriate shape.
    return cor.reshape(out_shape)
def covariance_map(pcs, field, ddof=1):
    """Covariance maps for a set of PCs and a spatial-temporal field.
    Given an array where the columns are PCs (e.g., as output from
    :py:meth:`eof2.EofSolve.pcs`) and an array containing a
    spatial-temporal where time is the first dimension, one covariance
    map per PC is computed.
    The field must have the same temporal dimension as the PCs. Any
    number of spatial dimensions (including zero) are allowed in the
    field and there can be any number of PCs.
    **Arguments:**
    *pcs*
        PCs as the columns of an array.
    *field*
        Spatial-temporal field with time as the first dimension.
    **Optional arguments:**
    *ddof*
        'Delta degrees of freedom'. The divisor used to normalize
        the covariance matrix is *N - ddof* where *N* is the
        number of samples. Defaults to *1*.
    """
    # Check PCs and fields for validity, flatten the arrays ready for the
    # computation and remove the mean along the leading dimension.
    pcs_cent, field_cent, out_shape = _check_flat_center(pcs, field)
    # Set the divisor according to the specified delta-degrees of freedom.
    div = np.float64(pcs_cent.shape[0] - ddof)
    # Compute the covariance map, making sure it has the appropriate shape.
    cov = (ma.dot(field_cent.T, pcs_cent).T / div).reshape(out_shape)
    return cov

class EofSolver(object):
    """EOF analysis (:py:mod:`numpy` interface)."""

    def __init__(self, dataset, weights=None, center=True, ddof=1):
        """Create an EofSolver object.
        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.
        **Arguments:**

        *dataset*
            A :py:class:`numpy.ndarray` or
            :py:class:`numpy.ma.core.MasekdArray` with two or more
            dimensions containing the data to be analysed. The first
            dimension is assumed to represent time. Missing values are
            permitted, either in the form of a masked array, or the
            value :py:attr:`numpy.nan`. Missing values must be constant
            with time (e.g., values of an oceanographic field over
            land).

        **Optional arguments:**
        *weights*
            An array of weights whose shape is compatible with those of
            the input array *dataset*. The weights can have the same
            shape as the input data set or a shape compatible with an
            array broadcast operation (ie. the shape of the weights can
            can match the rightmost parts of the shape of the input
            array *dataset*). If the input array *dataset* does not
            require weighting then the value *None* may be used.
            Defaults to *None* (no weighting).
        *center*
            If *True*, the mean along the first axis of the input data
            set (the time-mean) will be removed prior to analysis. If
            *False*, the mean along the first axis will not be removed.
            Defaults to *True* (mean is removed). Generally this option
            should be set to *True* as the covariance interpretation
            relies on input data being anomalies with a time-mean of 0.
            A valid reson for turning this off would be if you have
            already generated an anomaly data set. Setting to *True* has
            the useful side-effect of propagating missing values along
            the time-dimension, ensuring the solver will work even if
            missing values occur at different locations at different
            times.
        *ddof*
            'Delta degrees of freedom'. The divisor used to normalize
            the covariance matrix is *N - ddof* where *N* is the
            number of samples. Defaults to *1*.
        """
        # Store the input data in an instance variable.
        if dataset.ndim < 2:
            raise EofError("the input data set must be at least two dimensional")
        self._dataset = dataset.copy()
        # Check if the input is a masked array. If so fill it with NaN.
        try:
            self._dataset = self._dataset.filled(fill_value=np.nan)
            self._filled = True
        except AttributeError:
            self._filled = False
        # Store information about the shape/size of the input data.
        self._records = self._dataset.shape[0]
        self._originalshape = self._dataset.shape[1:]
        channels = np.product(self._originalshape)
        # Weight the data set according to weighting argument.
        if weights is not None:
            try:
                self._dataset = self._dataset * weights
                self._weights = weights
            except ValueError:
                raise EofError("weight array dimensions are incompatible")
            except TypeError:
                raise EofError("weights are not a valid type")
        else:
            self._weights = None
        # Remove the time mean of the input data unless explicitly told
        # not to by the "center" argument.
        self._centered = center
        if center:
            self._dataset = self._center(self._dataset)
        # Reshape to two dimensions (time, space) creating the design matrix.
        self._dataset = self._dataset.reshape([self._records, channels])
        # Find the indices of values that are not missing in one row. All the
        # rows will have missing values in the same places provided the
        # array was centered. If it wasn't then it is possible that some
        # missing values will be missed and the singular value decomposition
        # will produce not a number for everything.
        nonMissingIndex = np.where(np.isnan(self._dataset[0])==False)[0]
        # Remove missing values from the design matrix.
        dataNoMissing = self._dataset[:, nonMissingIndex]
        # Compute the singular value decomposition of the design matrix.
        A, Lh, E = np.linalg.svd(dataNoMissing, full_matrices=False)
        if np.any(np.isnan(A)):
            raise EofError("missing values encountered in SVD")
        # Singular values are the square-root of the eigenvalues of the
        # covariance matrix. Construct the eigenvalues appropriately and
        # normalize by N-ddof where N is the number of observations. This
        # corresponds to the eigenvalues of the normalized covariance matrix.
        self._ddof = ddof
        normfactor = float(self._records - self._ddof)
        self._L = Lh * Lh / normfactor
        # Store the number of eigenvalues (and hence EOFs) that were actually
        # computed.
        self.neofs = len(self._L)
        self._flatE = np.ones([self.neofs, channels],
                dtype=self._dataset.dtype) * np.NaN
        self._flatE = self._flatE.astype(self._dataset.dtype)
        self._flatE[:, nonMissingIndex] = E
        # Remove the scaling on the principal component time-series that is
        # implicitily introduced by using SVD instead of eigen-decomposition.
        # The PCs may be re-scaled later if required.
        self._P = A * Lh

    def _center(self, in_array):
        """Remove the mean of an array along the first dimension."""
        # Compute the mean along the first dimension.
        mean = in_array.mean(axis=0)
        # Return the input array with its mean along the first dimension
        # removed.
        return (in_array - mean)

    def pcs(self, pcscaling=0, npcs=None):
        """Principal component time series (PCs).
        
        Returns an array where the columns are the ordered PCs.
        
        **Optional arguments:**
        *pcscaling*
            Set the scaling of the retrieved PCs. The following
            values are accepted:
            * *0* : Un-scaled PCs (default).
            * *1* : PCs are scaled to unit variance (divided by the
              square-root of their eigenvalue).
            * *2* : PCs are multiplied by the square-root of their
              eigenvalue.
        *npcs* : Number of PCs to retrieve. Defaults to all the PCs.
        
        """
        slicer = slice(0, npcs)
        if pcscaling == 0:
            # Do not scale.
            return self._P[:, slicer].copy()
        elif pcscaling == 1:
            # Divide by the square-root of the eigenvalue.
            return self._P[:, slicer] / np.sqrt(self._L[slicer])
        elif pcscaling == 2:
            # Multiply by the square root of the eigenvalue.
            return self._P[:, slicer] * np.sqrt(self._L[slicer])
        else:
            raise EofError("invalid PC scaling option: %s" % repr(pcscaling))

 
    def eofs(self, eofscaling=0, neofs=None):
        """Empirical orthogonal functions (EOFs).
        
        Returns an array with the ordered EOFs along the first
        dimension.
        **Optional arguments:**
        *eofscaling*
            Sets the scaling of the EOFs. The following values are
            accepted:
            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their
              eigenvalues.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalues.
              
        *neofs* -- Number of EOFs to return. Defaults to all EOFs.
        
        """
        slicer = slice(0, neofs)
        neofs = neofs or self.neofs
        if eofscaling == 0:
            # No modification. A copy needs to be returned in case it is
            # modified. If no copy is made the internally stored eigenvectors
            # could be modified unintentionally.
            rval = self._flatE[slicer].copy()
        elif eofscaling == 1:
            # Divide by the square-root of the eigenvalues.
            rval = self._flatE[slicer] / np.sqrt(self._L[slicer])[:,_NA]
        elif eofscaling == 2:
            # Multiply by the square-root of the eigenvalues.
            rval = self._flatE[slicer] * np.sqrt(self._L[slicer])[:,_NA]
        else:
            raise EofError("invalid eof scaling option: %s" % repr(eofscaling))
        if self._filled:
            rval = ma.array(rval, mask=np.where(np.isnan(rval), True, False))
        return rval.reshape((neofs,) + self._originalshape)

    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EOF.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return. Defaults to all
            eigenvalues.

        """
        # Create a slicer and use it on the eigenvalue array. A copy must be
        # returned in case the slicer takes all elements, in which case a
        # reference to the eigenvalue array is returned. If this is modified
        # then the internal eigenvalues array would then be modified as well.
        slicer = slice(0, neigs)
        return self._L[slicer].copy()

    def eofsAsCorrelation(self, neofs=None):
        """
        EOFs scaled as the correlation of the PCs with the original
        field.

        **Optional argument:**

        *neofs*
            Number of EOFs to return. Defaults to all EOFs.

        """
        # Retrieve the specified number of PCs.
        pcs = self.pcs(npcs=neofs, pcscaling=1)
        # Compute the correlation of the PCs with the input field.
        c = correlation_map(pcs,
                self._dataset.reshape((self._records,)+self._originalshape))
        # The results of the correlation_map function will be a masked array.
        # For consistency with other return values, this is converted to a
        # numpy array filled with numpy.nan.
        if not self._filled:
            c = c.filled(fill_value=np.nan)
        return c

    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """
        EOFs scaled as the covariance of the PCs with the original
        field.
        **Optional arguments:**
        
        *neofs*
            Number of EOFs to return. Defaults to all EOFs.
        
        *pcscaling*
            Set the scaling of the PCs used to compute covariance. The
            following values are accepted:
            * *0* : Un-scaled PCs.
            * *1* : PCs are scaled to unit variance (divided by the
              square-root of their eigenvalue) (default).
            * *2* : PCs are multiplied by the square-root of their
              eigenvalue.
        """
        pcs = self.pcs(npcs=neofs, pcscaling=pcscaling)
        # Divide the input data by the weighting (if any) before computing
        # the covariance maps.
        data = self._dataset.reshape((self._records,)+self._originalshape)
        if self._weights is not None:
            with warnings.catch_warnings():
                # If any weight is zero this will produce a runtime error,
                # just ignore it.
                warnings.simplefilter('ignore', category=RuntimeWarning)
                data /= self._weights
        c = covariance_map(pcs, data, ddof=self._ddof)
        # The results of the covariance_map function will be a masked array.
        # For consitsency with other return values, this is converted to a
        # numpy array filled with numpy.nan.
        if not self._filled:
            c = c.filled(fill_value=np.nan)
        return c

    def varianceFraction(self, neigs=None):
        """Fractional EOF variances.
        
        The fraction of the total variance explained by each EOF. This
        is a value between 0 and 1 inclusive.
        **Optional argument:**
        *neigs*
            Number of eigenvalues to return the fractional variance for.
            Defaults to all eigenvalues.
        
        """
        # Return the array of eigenvalues divided by the sum of the
        # eigenvalues.
        slicer = slice(0, neigs)
        return self._L[slicer] / self._L.sum()

    def totalAnomalyVariance(self):
        """
        Total variance associated with the field of anomalies (the sum
        of the eigenvalues).
        
        """
        # Return the sum of the eigenvalues.
        return self._L.sum()

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.
        If weights were passed to the :py:class:`~eof2.EofSolver`
        instance then the returned reconstructed field will be
        automatically un-weighted. Otherwise the returned reconstructed
        field will  be weighted in the same manner as the input to the
        :py:class:`~eof2.EofSolver` instance.
        
        **Argument:**
        
        *neofs*
            Number of EOFs to use for the reconstruction.
        
        """
        # Project principal components onto the EOFs to compute the
        # reconstructed field.
        rval = np.dot(self._P[:, :neofs], self._flatE[:neofs])
        # Reshape the reconstructed field so it has the same shape as the
        # input data set.
        rval = rval.reshape((self._records,) + self._originalshape)
        # Un-weight the reconstructed field if weighting was performed on
        # the input data set.
        if self._weights is not None:
            rval = rval / self._weights
        # Return the reconstructed field.
        if self._filled:
            rval = ma.array(rval, mask=np.where(np.isnan(rval), True, False))
        return rval
    def northTest(self, neigs=None, vfscaled=False):
        """Typical errors for eigenvalues.
        
        The method of North et al. (1982) is used to compute the typical
        error for each eigenvalue. It is assumed that the number of
        times in the input data set is the same as the number of
        independent realizations. If this assumption is not valid then
        the result may be inappropriate.
        **Optional arguments:**
        
        *neigs*
            The number of eigenvalues to return typical errors for.
            Defaults to typical errors for all eigenvalues.
            
        *vfscaled*
            If *True* scale the errors by the sum of the eigenvalues.
            This yields typical errors with the same scale as the
            values returned by the
            :py:meth:`~eof2.EofSolver.varianceFraction` method. If
            *False* then no scaling is done. Defaults to *False* (no
            scaling).
        
        **References**
        North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng, 1982:
        "Sampling errors in the estimation of empirical orthogonal
        functions", *Monthly Weather Review*, **110**, pages 669-706.
        
        """
        slicer = slice(0, neigs)
        # Compute the factor that multiplies the eigenvalues. The number of
        # records is assumed to be the number of realizations of the field.
        factor = np.sqrt(2.0 / self._records)
        # If requested, allow for scaling of the eigenvalues by the total
        # variance (sum of the eigenvalues).
        if vfscaled:
            factor /= self._L.sum()
        # Return the typical errors.
        return self._L[slicer] * factor

    def getWeights(self):
        """Weights used for the analysis."""
        return self._weights

    def projectField(self, field, neofs=None, eofscaling=0, weighted=True):
        """Project a field onto the EOFs.

        Given a field, projects it onto the EOFs to generate a
        corresponding set of time series. The field can be projected
        onto all the EOFs or just a subset. The field must have the same
        corresponding spatial dimensions (including missing values in
        the same places) as the original input to the
        :py:class:`~eof2.EofSolver` instance. The field may have a
        different length time dimension to the original input field (or
        no time dimension at all).

        **Argument:**

        *field*
            A field to project onto the EOFs. The field should be
            contained in a :py:class:`numpy.ndarray` or a
            :py:class:`numpy.ma.core.MaskedArray`.
        **Optional arguments:**
        *neofs*
            Number of EOFs to project onto. Defaults to all EOFs.
        *eofscaling*
            Set the scaling of the EOFs that are projected
            onto. The following values are accepted:
            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their eigenvalue.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalue.
        *weighted*
            If *True* then the field is weighted prior to projection. If
            *False* then no weighting is applied. Defaults to *True*
            (weighting is applied). Generally only the default setting
            should be used.
        """
        # Check that the shape/dimension of the input field is compatible with
        # the EOFs.
        input_ndim = field.ndim
        eof_ndim = len(self._originalshape) + 1
        if eof_ndim - input_ndim not in (0, 1):
            raise EofError("field and EOFs have incompatible dimensions")
        # Check that the rightmost dimensions of the input field are the same as
        # the EOFs.
        if input_ndim == eof_ndim:
            check_shape = field.shape[1:]
        else:
            check_shape = field.shape
        if check_shape != self._originalshape:
            raise EofError("field and EOFs have incompatible shapes")
        # Create a slice object for truncating the EOFs.
        slicer = slice(0, neofs)
        # If required, weight the dataset with the same weighting that was
        # used to compute the EOFs.
        field = field.copy()
        if weighted:
            wts = self.getWeights()
            if wts is not None:
                field = field * wts
        # Fill missing values with NaN if this is a masked array.
        try:
            field = field.filled(fill_value=np.nan)
        except AttributeError:
            pass
        # Flatten the input field into [time, space] dimensionality.
        if eof_ndim > input_ndim:
            field = field.reshape((1,) + field.shape)
        records = field.shape[0]
        channels = np.product(field.shape[1:])
        field_flat = field.reshape([records, channels])
        # Locate the non-missing values and isolate them.
        nonMissingIndex = np.where(np.isnan(field_flat[0]) == False)[0]
        field_flat = field_flat[:, nonMissingIndex]
        # Locate the non-missing values in the EOFs and check they match those
        # in the input field, then isolate the non-missing values.
        eofNonMissingIndex = np.where(np.isnan(self._flatE[0]) == False)[0]
        if eofNonMissingIndex.shape != nonMissingIndex.shape or \
                (eofNonMissingIndex != nonMissingIndex).any():
            raise EofError("field and EOFs have different missing value locations")
        eofs_flat = self._flatE[slicer, eofNonMissingIndex]
        if eofscaling == 1:
            eofs_flat /= np.sqrt(self._L[slicer])[:,_NA]
        elif eofscaling == 2:
            eofs_flat *= np.sqrt(self._L[slicer])[:,_NA]
        # Project the field onto the EOFs using a matrix multiplication.
        projected_pcs = np.dot(field_flat, eofs_flat.T)
        if input_ndim < eof_ndim:
            # If an extra dimension was introduced, remove it before returning
            # the projected PCs.
            projected_pcs = projected_pcs[0]
        return projected_pcs


infile = '/scratch/gpfs/gn5970/data/GFDL-ESM4_tos_no3.nc'
data=xr.open_dataset(infile)
#data = data.dropna(dim='time', how='any')

print("data",data)
data['time']=pd.date_range("1850-01-01", periods=1980, freq="M")

min_lon = 0
min_lat = -90
min_depth = 0

max_lon = 360
max_lat = -50
max_depth = 50

mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
#mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data = data.where(mask_lat, drop=True)
#data=data.mean('depth')

lon=np.array(data.lon)
lat=np.array(data.lat)

no3_mean=data.no3.mean('time')*(1000)
no3_mean=np.array(no3_mean)
no3_std=data.no3.std('time')*(1000)
no3_std=np.array(no3_std)

for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()

    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(0, 32, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, no3_mean.squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("no3_mean_GFDL.png")



for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()

    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(0, 3.5, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, no3_std.squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("no3_std_GFDL.png")


temp_mean=data.tos.mean('time')
temp_std=data.tos.std('time')

infile = '/projects/CDEUTSCH/DATA/WOD23/wod_1955-2023.nc'
data=xr.open_dataset(infile)
print('data_ATLAS',data)

data['time']=pd.date_range("1955-01-01", periods=828, freq="M")
dtaa=data.isel(time=slice(100,828))
#data=data.mean('nbounds')
min_lon = -180
min_lat = -90
min_depth = 0

max_lon = 180
max_lat = -50
max_depth = 50

mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data = data.where(mask_lat & mask_depth, drop=True)
data=data.mean('depth')

lon=np.array(data.lon)
lat=np.array(data.lat)

no3_mean_atlas=data.no3.mean('time')
no3_mean_dd=data.no3_dd.mean('time')
print('no3_mean_atlas',no3_mean_atlas)
no3_mean_atlas=np.array(no3_mean_atlas)

no3_std_atlas=data.no3.std('time')
print('no3_mean_atlas',no3_std_atlas)
no3_std_atlas=np.array(no3_std_atlas)

no3_mean_dd=np.array(no3_mean_dd)
for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()

    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(0, 32, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, no3_mean_atlas.squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    ax.contour(lon,lat, no3_mean_atlas.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("no3_WODB.png")

for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()

    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(0, 3, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, no3_std_atlas.squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    ax.contour(lon,lat, no3_std_atlas.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("no3_std_WODB.png")




mask = np.abs(no3_mean-no3_mean_atlas) < 6

# Select the points where the difference is less than 1
filtered_difference = np.where(mask, no3_mean-no3_mean_atlas, np.nan)

for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(-6, 6, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, (no3_mean-no3_mean_atlas).squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    ax.contour(lon,lat, (no3_mean-no3_mean_atlas).squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("no3_bias.png")


for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(0, 1, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, filtered_difference.squeeze(), cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("no3_bias_onlyvalid.png")



mask = np.abs(no3_std-no3_std_atlas) < 1

# Select the points where the difference is less than 1
filtered_difference = np.where(mask, no3_std-no3_std_atlas, np.nan)

for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()
    # Define the yticks for latitude
    #lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(0, 0.7, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, filtered_difference.squeeze(), cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())

    #ax.contour(lon,lat, no3_mean.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig("no3_bias_std_onlyvalid.png")


# Print the results
#print(f"T-statistic: {t_stat}")
#print(f"P-value: {p_value}")

# Interpret the p-value
#alpha = 0.05  # Significance level
#if p_value < alpha:
#    print("The residuals are statistically significant (reject H0).")
#else:
#    print("The residuals are not statistically significant (fail to reject H0).")



#infile = '/scratch/gpfs/gn5970/data/GFDL-ESM4_tos_no3.nc'
infile = '/projects/CDEUTSCH/DATA/WOD23/wod_1955-2023_anom.nc'
#data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data=xr.open_dataset(infile)
data['time']=pd.date_range("1955-01-01", periods=828, freq="M")
min_lon = -180
min_lat = -90
min_depth = 0

max_lon = 180
max_lat = -50
max_depth = 50

mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data = data.where(mask_lat & mask_depth, drop=True)
data=data.mean('depth')
#data_no3=np.array(detrend_dim(data.no3,dim='time'))
lon=np.array(data.lon)
lat=np.array(data.lat)
print('data_GFDL',data)

#lon=np.array(selected_values.lon.data)

#print('lat',lat)
#print('lon',lon)


from scipy.signal import detrend




#lat=np.array(data.lat.data)
#lon=np.array(data.lon.data)
# temperature weighted by grid-cell area
no3=data.no3
#no3_clim = no3.groupby('time.month').mean(dim='time',skipna=True)
#no3_anom = no3.groupby('time.month') - no3_clim
#sst_anom=sst_anom.coarsen(time=3).mean()
#no3_weighted =data.no3#(data.tos*area) / total_area
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

#def detrend(data, axis=-1):
#    """Remove linear trend along axis"""
#    p = np.polyfit(np.arange(data.shape[axis]), data, 1, axis=axis)
#    return data - np.polyval(p, np.arange(data.shape[axis])[..., None], axis=axis)

def wgt_areaave(indat, latS, latN, lonW, lonE):
    lat = indat.lat
    lon = indat.lon

    # Handle longitude wrapping
    if ((lonW < 0) or (lonE < 0)) and (lon.values.min() > -1):
        anm = indat.assign_coords(lon=((lon + 180) % 360 - 180))
        lon = (lon + 180) % 360 - 180
    else:
        anm = indat

    iplat = lat.where((lat >= latS) & (lat <= latN), drop=True)
    iplon = lon.where((lon >= lonW) & (lon <= lonE), drop=True)

    # Mask NaN values
    valid_mask = ~np.isnan(anm)

    # Calculate weights using the cosine of latitude
    wgt = np.cos(np.deg2rad(lat))

    # Apply the weights and calculate the mean, excluding NaN values
    odat = anm.where(valid_mask).sel(lat=iplat, lon=iplon).weighted(wgt).mean(("lon", "lat"), skipna=True)
   

    return odat


anomaly_no3 = wgt_areaave(no3, -90, -50, -180, 180)
anomaly_no3 = anomaly_no3 / anomaly_no3.std()
no3_glb_avg=np.array(anomaly_no3)
# Calculate z-scores and exclude outliers
mean = np.nanmean(no3_glb_avg)
standard_deviation = np.nanstd(no3_glb_avg)
distance_from_mean = abs(no3_glb_avg - mean)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
outlier=distance_from_mean > max_deviations * standard_deviation
no3_glb_avg=no3_glb_avg[not_outlier]
no3_glb_avg=detrend_dim(xr.DataArray(no3_glb_avg),dim='dim_0')


#no3_index=np.array(no3_index)
print('no3_glb_avg',no3_glb_avg)

plt.figure()
no3_glb_avg.coarsen(dim_0=3).mean().plot()
ax = plt.gca()
ax.set_title('')  # Set an empty string as the title
#plt.xticks(range(1850,2014,10), rotation='vertical')
#plt.axhline(0, color='k')
plt.xlabel('Year',fontsize=20)
plt.ylabel('NO3 anomalies',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("nitrogen_anomaly_GFDL2.pdf")

no3_index=np.array(no3_glb_avg)
#print('non_nan_values_list',non_nan_values_list.shape)

anomaly_tos = wgt_areaave(data.temp, -90, -50, -180, 180)
anomaly_tos = anomaly_tos / anomaly_tos.std()
tos_glb_avg=np.array(anomaly_tos)
# Calculate z-scores and exclude outliers
mean = np.nanmean(tos_glb_avg)
standard_deviation = np.nanstd(tos_glb_avg)
distance_from_mean = abs(tos_glb_avg - mean)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
outlier=distance_from_mean > max_deviations * standard_deviation
tos_glb_avg=tos_glb_avg[not_outlier]
tos_glb_avg=detrend_dim(xr.DataArray(tos_glb_avg),dim='dim_0')


#print("A",A_tos.shape)

year=np.arange(1850,2015,1/12)
plt.figure()
tos_glb_avg.isel(dim_0=slice(0,690)).coarsen(dim_0=3).mean().plot()
plt.xlabel('Year',fontsize=20)
#plt.ylabel('NPP anomalies',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('tos_index.pdf')

anomaly_salt = wgt_areaave(data.salt, -90, -50, -180, 180)
anomaly_salt = anomaly_salt / anomaly_salt.std()
salt_glb_avg=np.array(anomaly_salt)
# Calculate z-scores and exclude outliers
mean = np.nanmean(salt_glb_avg)
standard_deviation = np.nanstd(salt_glb_avg)
distance_from_mean = abs(salt_glb_avg - mean)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
outlier=distance_from_mean > max_deviations * standard_deviation
salt_glb_avg=salt_glb_avg[not_outlier]
salt_glb_avg=detrend_dim(xr.DataArray(salt_glb_avg),dim='dim_0')


#print("A",A_tos.shape)

year=np.arange(1850,2015,1/12)
plt.figure()
salt_glb_avg.isel(dim_0=slice(0,678)).coarsen(dim_0=3).mean().plot()
plt.xlabel('Year',fontsize=20)
#plt.ylabel('NPP anomalies',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('salt_index.pdf')



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



# older code
infile = '/scratch/gpfs/gn5970/data/GFDL-ESM4_tos_no3.nc'
#data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data=xr.open_dataset(infile)
import datetime
data['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
min_lon = 0
min_lat = -90
min_depth = 0

max_lon = 360
max_lat = -50
max_depth = 50

mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
#mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data = data.where(mask_lat , drop=True)
lon=np.array(data.lon)
lat=np.array(data.lat)

tos_weighted =data.tos#(data.so*area) / total_area

tos_detrend=detrend_dim(tos_weighted,dim='time')

tos_clim = tos_detrend.groupby('time.month').mean(dim='time',skipna=True)
tos_anom = tos_detrend.groupby('time.month') - tos_clim
#so_anom=so_anom.coarsen(time=3).mean()
#tos_anom=tos_anom.coarsen(time=2).mean()
tos_anom2=tos_anom/tos_anom.std()
A_tos2=np.array(tos_anom2)

tos_anom=wgt_areaave(tos_anom,-90,-50,0,360)
tos_index=tos_anom/tos_anom.std()
tos_index=np.array(tos_index)

no3_weighted =data.no3#(data.so*area) / total_area

no3_detrend=detrend_dim(no3_weighted,dim='time')

no3_clim = no3_detrend.groupby('time.month').mean(dim='time',skipna=True)
no3_anom = no3_detrend.groupby('time.month') - no3_clim

#no3_anom=no3_anom.coarsen(time=2).mean()
#so_anom=so_anom.coarsen(time=3).mean()
no3_anom2=no3_anom/no3_anom.std()
A_no3=np.array(no3_anom2)

no3_anom=wgt_areaave(no3_anom,-90,-50,0,360)
no3_index=no3_anom/no3_anom.std()


year=np.arange(1850,2015,1/12)
plt.figure()
no3_index.plot()
plt.xlabel('Year',fontsize=20)
#plt.ylabel('NPP anomalies',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('no3_index2.pdf')


correlation_no3_tos=[]
correlation_no3_no3=[]
for i in range(40):
    for j in range(360):
        data_subset_no3 = no3_anom2.isel(lat=i,lon=j, drop=True)
        data_subset_tos = tos_anom2.isel(lat=i,lon=j, drop=True)

        correlation_no3_tos.append(xr.corr(no3_index,data_subset_tos))
        correlation_no3_no3.append(xr.corr(no3_index,data_subset_no3))

correlation_no3_tos=np.array(correlation_no3_tos)
correlation_no3_tos=correlation_no3_tos.reshape(40,360)

correlation_no3_no3=np.array(correlation_no3_no3)
correlation_no3_no3=correlation_no3_no3.reshape(40,360)


print('correlation',correlation_no3_tos.shape)

#no3_index=np.array(no3_index)




for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()

    #T_corr=correlation_map(no3_index,A_tos2[:,:,:])
    #T_reg=np.divide(covariance_map(no3_index, A_tos2[:,:,:]),np.var(no3_index))

    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(-0.6, 0.6, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, correlation_no3_tos.squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    df = 40
    sig=xr.DataArray(data=correlation_no3_tos*np.sqrt((df-2)/(1-np.square(correlation_no3_tos))),
      dims=["lat","lon'"],
      coords=[lat, lon])
    t90=stats.t.ppf(1-0.05, df-2)
    t95=stats.t.ppf(1-0.025, df-2)
    sig.plot.contourf(ax=ax,levels = [-1*t95, -1*t90, t90, t95], colors='none',
      hatches=['..', None, None, None, '..'], extend='both',
      add_colorbar=False, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
plt.savefig("correlation_tos_no3.pdf")

max_lag = 60  # Define the maximum lag you want to consider
lags, correlations, p_values = lagged_correlation(no3_index, tos_index, max_lag)

# Convert to pandas DataFrame for better visualization
correlation_df = pd.DataFrame({'Lag': lags, 'Correlation': correlations, 'p-value': p_values})

# Plot the lagged correlation
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(correlation_df['Lag'], correlation_df['Correlation'], marker='o', label='Correlation')
plt.axhline(0, color='grey', linestyle='--')
plt.axhline(y=0.196, color='red', linestyle='--', label='Significance Level (p < 0.05)')
plt.axhline(y=-0.196, color='red', linestyle='--')
plt.grid(True)

# Highlight significant correlations
significance_level = 0.05
significant = correlation_df['p-value'] < significance_level
plt.scatter(correlation_df['Lag'][significant], correlation_df['Correlation'][significant], color='red', label='Significant (p < 0.05)')
plt.xlabel("Lag",fontsize=18, fontweight='bold')
plt.ylabel("Correlation",fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
plt.show()
plt.savefig('no3_tos_lag.pdf')

# Create a figure and axis with polar stereographic projection
fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=1200, subplot_kw={'projection': ccrs.SouthPolarStereo()})

# Add coastlines and gridlines
ax.coastlines(resolution='110m')
ax.gridlines()

# Define contour levels
v = np.linspace(-0.8, 0.8, 40, endpoint=True)

# Contour plot with PlateCarree projection
fill = ax.contourf(lon, lat, correlation_no3_no3.squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

# Adjust aspect ratio for a circular plot
ax.set_aspect('equal')

# Add colorbar
cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)

# Significance testing
df = 40
sig = xr.DataArray(data=correlation_no3_no3 * np.sqrt((df - 2) / (1 - np.square(correlation_no3_no3))),
                   dims=["lat", "lon"],
                   coords=[lat, lon])
t90 = stats.t.ppf(1 - 0.05, df - 2)
t95 = stats.t.ppf(1 - 0.025, df - 2)

# Plot significance contours
sig.plot.contourf(ax=ax, levels=[-t95, -t90, t90, t95], colors='none',
                  hatches=['..', None, None, '..'], extend='both', 
                  add_colorbar=False, transform=ccrs.PlateCarree())

# Set extent of the plot
ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())

plt.savefig("correlation_no3_no3.pdf")




infile = '/scratch/gpfs/gn5970/data/GFDL-ESM4_so_zos.nc'
#data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data=xr.open_dataset(infile)
print('data_so',data)
import datetime
data['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
data['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
min_lon = 0
min_lat = -90
min_depth = 0

max_lon = 360
max_lat = -50
max_depth = 50

mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
#mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data = data.where(mask_lat , drop=True)
lon=np.array(data.lon)
lat=np.array(data.lat)


so_mean_GFDL=np.array(data.so.mean('time'))


lat=data.lat
lon=data.lon
time=data.time



lat=np.array(data.lat)
lon=np.array(data.lon)
time=np.array(data.time)

# temperature weighted by grid-cell area
so_weighted =data.so#(data.so*area) / total_area

so_detrend=detrend_dim(so_weighted,dim='time')

so_clim = so_detrend.groupby('time.month').mean(dim='time',skipna=True)
so_anom = so_detrend.groupby('time.month') - so_clim
#so_anom=so_anom.coarsen(time=3).mean()
#so_anom=so_anom.coarsen(time=2).mean()
so_anom2=so_anom/so_anom.std()
A_so2=np.array(so_anom2)

so_anom=wgt_areaave(so_anom,-90,-50,0,360)
so_index=so_anom/so_anom.std()
so_index=np.array(so_index)


correlation_no3_so=[]
for i in range(40):
    for j in range(360):
        #data_subset_no3 = no3_anom2.isel(lat=i,lon=j, drop=True)
        data_subset_so = so_anom2.isel(lat=i,lon=j, drop=True)

        correlation_no3_so.append(xr.corr(no3_index,data_subset_so))

correlation_no3_so=np.array(correlation_no3_so)
correlation_no3_so=correlation_no3_so.reshape(40,360)
print('correlation',correlation_no3_so.shape)


max_lag = 60  # Define the maximum lag you want to consider
lags, correlations, p_values = lagged_correlation(no3_index, so_index, max_lag)

# Convert to pandas DataFrame for better visualization
correlation_df = pd.DataFrame({'Lag': lags, 'Correlation': correlations, 'p-value': p_values})

# Plot the lagged correlation
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(correlation_df['Lag'], correlation_df['Correlation'], marker='o', label='Correlation')
plt.axhline(0, color='grey', linestyle='--')
plt.axhline(y=0.196, color='red', linestyle='--', label='Significance Level (p < 0.05)')
plt.axhline(y=-0.196, color='red', linestyle='--')
plt.grid(True)

# Highlight significant correlations
significance_level = 0.05
significant = correlation_df['p-value'] < significance_level
plt.scatter(correlation_df['Lag'][significant], correlation_df['Correlation'][significant], color='red', label='Significant (p < 0.05)')
plt.ylabel("Correlation",fontsize=18, fontweight='bold')
plt.xlabel("Lag",fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
plt.show()
plt.savefig('no3_so_lag.pdf')




for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()

    #T_corr=correlation_map(no3_index,A_tos2[:,:,:])
    #T_reg=np.divide(covariance_map(no3_index, A_tos2[:,:,:]),np.var(no3_index))

    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(-0.6, 0.6, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon, lat, correlation_no3_so.squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    df = 40
    sig=xr.DataArray(data=correlation_no3_so*np.sqrt((df-2)/(1-np.square(correlation_no3_so))),
      dims=["lat","lon"],
      coords=[lat, lon])
    t90=stats.t.ppf(1-0.05, df-2)
    t95=stats.t.ppf(1-0.025, df-2)
    sig.plot.contourf(ax=ax,levels = [-1*t95, -1*t90, t90, t95], colors='none',
      hatches=['..', None, None, None, '..'], extend='both',
      add_colorbar=False, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
plt.savefig("correlation_so_no3.pdf")




lat=np.array(data.lat)
lon=np.array(data.lon)
time=np.array(data.time)

zos_weighted =data.zos#(data.zos*area) / total_area

zos_detrend=detrend_dim(zos_weighted,dim='time')
zos_clim = zos_detrend.groupby('time.month').mean(dim='time',skipna=True)
zos_anom = zos_detrend.groupby('time.month') - zos_clim
#zos_anom=zos_anom.coarsen(time=2).mean()
zos_anom2=zos_anom/zos_anom.std()

A_zos=np.array(zos_anom2)
zos_anom=wgt_areaave(zos_anom,-90,-50,0,360)
zos_index=zos_anom/zos_anom.std()

year=np.arange(1850,2015,1/12)
plt.figure()
zos_index.plot()
plt.xlabel('Year',fontsize=20)
#plt.ylabel('NPP anomalies',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('zos_index.pdf')



max_lag = 60  # Define the maximum lag you want to consider
lags, correlations, p_values = lagged_correlation(no3_index, zos_index, max_lag)

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
plt.grid(True)
# Highlight significant correlations
significance_level = 0.05
significant = correlation_df['p-value'] < significance_level
plt.scatter(correlation_df['Lag'][significant], correlation_df['Correlation'][significant], color='red', label='Significant (p < 0.05)')

plt.legend()
plt.show()
plt.savefig('no3_zos_lag.pdf')


infile = '/scratch/gpfs/gn5970/data/GFDL-ESM4-siconc.nc'
#data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data=xr.open_dataset(infile)
print('data_siconc',data)
import datetime
data['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
data['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
min_lon = 0
min_lat = -90
min_depth = 0

max_lon = 360
max_lat = -50
max_depth = 50

mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
#mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data = data.where(mask_lat , drop=True)
lon=np.array(data.lon)
lat=np.array(data.lat)


#siconc_mean_GFDL=np.array(data.siconc.mean('time'))
siconc_weighted =data.siconc#(data.zos*area) / total_area

siconc_detrend=detrend_dim(siconc_weighted,dim='time')
siconc_clim = siconc_detrend.groupby('time.month').mean(dim='time',skipna=True)
siconc_anom = siconc_detrend.groupby('time.month') - siconc_clim
#zos_anom=zos_anom.coarsen(time=2).mean()
siconc_anom2=siconc_anom/siconc_anom.std()

A_siconc=np.array(siconc_anom2)
siconc_anom=wgt_areaave(siconc_anom,-90,-50,0,360)
siconc_index=siconc_anom/siconc_anom.std()

max_lag = 60  # Define the maximum lag you want to consider
lags, correlations, p_values = lagged_correlation(no3_index,siconc_index, max_lag)

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
plt.ylabel("Correlation",fontsize=18)
plt.xlabel("Lag",fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
plt.show()
plt.savefig('no3_si_lag.pdf')



correlation_no3_zos=[]
for i in range(40):
    for j in range(360):
        data_subset_no3 = no3_anom2.isel(lat=i,lon=j, drop=True)
        data_subset_zos = zos_anom2.isel(lat=i,lon=j, drop=True)

        correlation_no3_zos.append(xr.corr(no3_index,data_subset_zos))

correlation_no3_zos=np.array(correlation_no3_zos)
correlation_no3_zos=correlation_no3_zos.reshape(40,360)
print('correlation',correlation_no3_zos.shape)





for i in range(1):
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)

    # Create a polar stereographic projection
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    ax.coastlines(resolution='110m')
    ax.gridlines()

    #T_corr=correlation_map(no3_index,A_tos2[:,:,:])
    #T_reg=np.divide(covariance_map(no3_index, A_tos2[:,:,:]),np.var(no3_index))

    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    #ax.yaxis.set_major_formatter(lat_formatter)

    v = np.linspace(-0.6, 0.6, 40, endpoint=True)

    # Contour plot with PlateCarree projection
    fill = ax.contourf(lon-180, lat, correlation_no3_zos.squeeze(), v, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())

    # Make the aspect ratio equal to get a circular plot
    ax.set_aspect('equal')

    # Colorbar
    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    font_size = 20  # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    # Set the latitude limits
    df = 40
    sig=xr.DataArray(data=correlation_no3_zos*np.sqrt((df-2)/(1-np.square(correlation_no3_zos))),
      dims=["lat","lon"],
      coords=[lat, lon-180])
    t90=stats.t.ppf(1-0.05, df-2)
    t95=stats.t.ppf(1-0.025, df-2)
    sig.plot.contourf(ax=ax,levels = [-1*t95, -1*t90, t90, t95], colors='none',
      hatches=['..', None, None, None, '..'], extend='both',
      add_colorbar=False, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
plt.savefig("correlation_zos_no3.pdf")



zos_index=np.array(zos_index)
no3_index=np.array(no3_index)

infile = '/scratch/gpfs/gn5970/mld-GFDL-ESM4.nc'
#data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data=xr.open_dataset(infile)


import datetime
data['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
min_lon = 0
min_lat = -90
min_depth = 0

max_lon = 360
max_lat = -50
max_depth = 50

mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
#mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data = data.where(mask_lat , drop=True)


lat=np.array(data.lat)
lon=np.array(data.lon)

mld_weighted =data.mlotst#(data.mlotst*area) / total_area
mld_detrend=detrend_dim(mld_weighted,dim='time')

anomalies= mld_detrend.groupby('time.month')-mld_detrend.groupby('time.month').mean('time',skipna=True)
anomalies2=anomalies/anomalies.std()
#anomalies=anomalies.coarsen(time=3).mean()

#print("anomalies",anomalies.shape)

#anomalies=anomalies.reshape(720,60,360)
#anomalies=anomalies.transpose('variable','time','lat','lon')
#anomalies=anomalies.mean('variable')
from eofs.xarray import Eof

A_mld=np.array(anomalies2)

mld_weighted =data.mlotst#(data.mlotst*area) / total_area
mld_detrend=detrend_dim(mld_weighted,dim='time')

anomalies= mld_detrend.groupby('time.month')-mld_detrend.groupby('time.month').mean('time',skipna=True)

anomalies=wgt_areaave(anomalies,-90,-50,0,360)
mld_index=anomalies#.coarsen(time=2).mean()
mld_index=mld_index/mld_index.std()

year=np.arange(1850,2015,1/12)
plt.figure()
mld_index.plot()
plt.xlabel('Year',fontsize=20)
#plt.ylabel('NPP anomalies',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('mld_index.pdf')


max_lag = 60  # Define the maximum lag you want to consider
lags, correlations, p_values = lagged_correlation(no3_index, mld_index, max_lag)

# Convert to pandas DataFrame for better visualization
correlation_df = pd.DataFrame({'Lag': lags, 'Correlation': correlations, 'p-value': p_values})

# Plot the lagged correlation
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(correlation_df['Lag'], correlation_df['Correlation'], marker='o', label='Correlation')
plt.axhline(0, color='grey', linestyle='--')
plt.axhline(y=0.196, color='red', linestyle='--', label='Significance Level (p < 0.05)')
plt.axhline(y=-0.196, color='red', linestyle='--')

# Highlight significant correlations
plt.grid(True)
significance_level = 0.05
significant = correlation_df['p-value'] < significance_level
plt.scatter(correlation_df['Lag'][significant], correlation_df['Correlation'][significant], color='red', label='Significant (p < 0.05)')
plt.xlabel("Lag",fontsize=18, fontweight='bold')
plt.ylabel("Correlation",fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
plt.show()
plt.savefig('no3_mld_lag.pdf')




infile = '/projects/CDEUTSCH/DATA/ocean_monthly_z.static.nc'
ds=xr.open_dataset(infile)
grid_model = xr.Dataset()
grid_model['lon'] = ds['geolon_c']
grid_model['lat'] = ds['geolat_c']
print("grid",ds)

min_lat = -90
max_lat =  -50

infile = '/projects/CDEUTSCH/DATA/tauuo-GFDL-ESM4.nc'
#data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
taux=xr.open_dataset(infile)
taux=taux.tauuo
taux['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
#min_lat = -90

#max_lat =  -30

#mask_lat = (taux.y >= min_lat) & (taux.y <= max_lat)

#taux = taux.where(mask_lat, drop=True)

lon_wsc=np.array(taux.x)
lat_wsc=np.array(taux.y)

infile = '/projects/CDEUTSCH/DATA/tauvo-GFDL-ESM4.nc'
#data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
tauy=xr.open_dataset(infile)
tauy=tauy.tauvo
tauy['time']=pd.date_range("1850-01-01", periods=1980, freq="M")

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


rho0 = 1028


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
lon_ep=ekman_pumping.lon
lat_ep=ekman_pumping.lat

ekman_pumping_weighted=detrend_dim(ekman_pumping,dim='time')
ekman_pumping_clim = ekman_pumping_weighted.groupby('time.month').mean(dim='time',skipna=True)
ekman_pumping_anom = ekman_pumping_weighted.groupby('time.month') - ekman_pumping_clim
ekman_pumping_anom=ekman_pumping_anom/ekman_pumping_anom.std()
A_ekman_pumping=np.array(ekman_pumping_anom)



ekman_pumping_weighted=detrend_dim(ekman_pumping,dim='time')
ekman_pumping_clim = ekman_pumping_weighted.groupby('time.month').mean(dim='time',skipna=True)
ekman_pumping_anom = ekman_pumping_weighted.groupby('time.month') - ekman_pumping_clim
ekman_pumping_index=wgt_areaave(ekman_pumping_anom,-90,-50,0,360)
#ekman_pumping_index=ekman_pumping_index.coarsen(time=2).mean()
ekman_pumping_index=ekman_pumping_index/ekman_pumping_index.std()

year=np.arange(1850,2015,1/12)
plt.figure(figsize=(10, 10),dpi=1200)
ekman_pumping_index.plot()
plt.xlabel('Year',fontsize=20)
#plt.ylabel('NPP anomalies',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('ek_index.pdf')


max_lag = 60  # Define the maximum lag you want to consider
lags, correlations, p_values = lagged_correlation(no3_index, ekman_pumping_index, max_lag)

# Convert to pandas DataFrame for better visualization
correlation_df = pd.DataFrame({'Lag': lags, 'Correlation': correlations, 'p-value': p_values})

# Plot the lagged correlation
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(correlation_df['Lag'], correlation_df['Correlation'], marker='o', label='Correlation')
plt.axhline(0, color='grey', linestyle='--')
plt.axhline(y=0.196, color='red', linestyle='--', label='Significance Level (p < 0.05)')
plt.axhline(y=-0.196, color='red', linestyle='--')

plt.grid(True)

# Highlight significant correlations
significance_level = 0.05
significant = correlation_df['p-value'] < significance_level
plt.scatter(correlation_df['Lag'][significant], correlation_df['Correlation'][significant], color='red', label='Significant (p < 0.05)')
plt.ylabel("Correlation",fontsize=18, fontweight='bold')
plt.xlabel("Lag",fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
plt.show()
plt.savefig('no3_ep_lag.pdf')


infile = '/scratch/gpfs/gn5970/data/GFDL-ESM4_tos_no3.nc'
#data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data=xr.open_dataset(infile)
import datetime

data['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
data['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
min_lon = 0
min_lat = -90
min_depth = 0

max_lon = 360
max_lat = -50
max_depth = 50

mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
#mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data = data.where(mask_lat , drop=True)



infile = '/scratch/gpfs/gn5970/data/GFDL-ESM4_so_zos.nc'
#data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data2=xr.open_dataset(infile)
import datetime
data2['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
min_lon = 0
min_lat = -90
min_depth = 0

max_lon = 360
max_lat = -50
max_depth = 50

mask_lon = (data2.lon >= min_lon) & (data2.lon <= max_lon)
mask_lat = (data2.lat >= min_lat) & (data2.lat <= max_lat)
#mask_depth = (data.depth >= min_depth) & (data.depth <= max_depth)
data2 = data2.where(mask_lat , drop=True)



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

pt = xr.apply_ufunc(pdens, data2.so, data.tos,
                    dask='parallelized',
                    output_dtypes=[data2.so.dtype])

rho_ref = 1035.
anom_density = pt - rho_ref

g = 9.81
buoyancy = -g * anom_density / rho_ref

#buoyancy=(buoyancy*area)/ total_area
b_detrend=detrend_dim(buoyancy,dim='time')
b_clim = b_detrend.groupby('time.month').mean(dim='time',skipna=True)
b_anom = b_detrend.groupby('time.month') - b_clim
#zos_anom=zos_anom.coarsen(time=3).mean()
b_anom2=b_anom/b_anom.std()



A_b=np.array(b_anom2)

b_anom=wgt_areaave(b_anom,-90,-50,0,360)
#b_anom=b_anom.coarsen(time=2).mean()
b_index=b_anom/b_anom.std()

year=np.arange(1850,2015,1/12)
plt.figure(figsize=(10, 10),dpi=1200)
b_index.plot()

plt.savefig('b_index.pdf')

b_index=np.array(b_index)
max_lag = 60  # Define the maximum lag you want to consider
lags, correlations, p_values = lagged_correlation(no3_index, b_index, max_lag)

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
plt.grid(True)

# Highlight significant correlations
significance_level = 0.05
significant = correlation_df['p-value'] < significance_level
plt.scatter(correlation_df['Lag'][significant], correlation_df['Correlation'][significant], color='red', label='Significant (p < 0.05)')
plt.ylabel("Correlation",fontsize=18, fontweight='bold')
plt.xlabel("Lag",fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
plt.show()

plt.savefig('no3_b_lag.pdf')





pt=pt#(pt*area)/total_area
pd_detrend=detrend_dim(pt,dim='time')
pd_clim = pd_detrend.groupby('time.month').mean(dim='time')
pd_anom = pd_detrend.groupby('time.month') - pd_clim

#zos_anom=zos_anom.coarsen(time=3).mean(
pd_anom2=wgt_areaave(pd_anom,-90,-50,0,360)
#pd_anom2=pd_anom2.coarsen(time=2).mean()
pd_index=pd_anom2/pd_anom2.std()

year=np.arange(1850,2015,1/12)
plt.figure()
pd_index.plot()
plt.xlabel('Year',fontsize=20)
#plt.ylabel('NPP anomalies',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('pd_index.pdf')


lags, correlations, p_values = lagged_correlation(no3_index, pd_index, max_lag)

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
#plt.title('Lagged Correlation between NO3 and MLD')
plt.grid(True)

# Highlight significant correlations
significance_level = 0.05
significant = correlation_df['p-value'] < significance_level
plt.scatter(correlation_df['Lag'][significant], correlation_df['Correlation'][significant], color='red', label='Significant (p < 0.05)')
plt.ylabel("Correlation",fontsize=18, fontweight='bold')
plt.xlabel("Lag",fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
plt.show()
plt.savefig('no3_pd_lag.pdf')


pd_index=np.array(pd_index)

pd_anom3=pd_anom/pd_anom.std()

A_pd=np.array(pd_anom3)


infile = '/projects/CDEUTSCH/DATA/GFDL-ESM4-CMIP-fe.nc'
#data = xr.open_mfdataset(infile, drop_variables=['time_bnds'])
data=xr.open_dataset(infile)

data['time']=pd.date_range("1850-01-01", periods=1980, freq="M")
fe=data.dfeos

min_lat = -90

max_lat =  -50

mask_lat = (fe.lat >= min_lat) & (fe.lat <= max_lat)

fe = fe.where(mask_lat, drop=True)

#fe=(fe*area)/ total_area
fe_detrend=detrend_dim(fe,dim='time')
fe_clim = fe_detrend.groupby('time.month').mean(dim='time',skipna=True)
fe_anom = fe_detrend.groupby('time.month') - fe_clim
#fe_anom=fe_anom.coarsen(time=2).mean()
#zos_anom=zos_anom.coarsen(time=3).mean()
fe_anom2=fe_anom/fe_anom.std()

A_fe=np.array(fe_anom2)

fe_anom=wgt_areaave(fe_anom,-90,-50,0,360)
fe_index=fe_anom/fe_anom.std()

year=np.arange(1850,2015,1/12)
plt.figure()
fe_index.plot()
plt.xlabel('Year',fontsize=20)
#plt.ylabel('NPP anomalies',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('fe_index.pdf')


fe_index=np.array(fe_index)
max_lag = 60  # Define the maximum lag you want to consider
lags, correlations, p_values = lagged_correlation(no3_index, fe_index, max_lag)

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
plt.title('Lagged Correlation between NO3 and MLD')
plt.grid(True)

# Highlight significant correlations
significance_level = 0.05
significant = correlation_df['p-value'] < significance_level
plt.scatter(correlation_df['Lag'][significant], correlation_df['Correlation'][significant], color='red', label='Significant (p < 0.05)')
plt.ylabel("Correlation",fontsize=13, fontweight='bold')
plt.xlabel("Lag",fontsize=13, fontweight='bold')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()
plt.show()
plt.savefig('no3_fe_lag.pdf')



def ts_diff(ts):
    diff_ts = [0] * len(ts)
    for i in range(1, len(ts)):
        diff_ts[i] = ts[i] - ts[i - 1]
    return diff_ts

def ts_int(ts_diff, ts_base, start=0):
        """
        Integrate a differenced time series using cumulative sum.

        Parameters:
        - ts_diff (numpy array): The differenced time series.
        - ts_base (numpy array): The base time series.
        - start (float): The initial value for integration.

        Returns:
        - ts_integrated (numpy array): The integrated time series.
        """
        ts_diff = np.asarray(ts_diff)
        ts_base = np.asarray(ts_base)

        ts_integrated = np.empty_like(ts_diff)
        ts_integrated[0] = start + ts_diff[0]

        # Use cumulative sum for integration
        ts_integrated[1:] = np.cumsum(ts_diff[1:]) + ts_base[:-1]
        return ts_integrated.tolist()

#def ts_int(ts_diff, ts_base, start=0):
#        ts_diff = np.asarray(ts_diff)  # Convert to NumPy array for vectorized operations
#        ts_base = np.asarray(ts_base)

#        ts = np.empty_like(ts_diff)  # Create an empty array to store the integrated time series
#        ts[0] = start + ts_diff[0]  # Set the initial value

        # Perform vectorized addition to calculate the integrated series
#        ts[1:] = ts_diff[1:] + ts_base[:-1]

#        return ts.tolist()

#        ts_diff = np.asarray(ts_diff)  # Convert to NumPy array for vectorized operations
#        ts_base = np.asarray(ts_base)

#        ts = np.empty_like(ts_diff)  # Create an empty array to store the integrated time series
#        ts[0] = start + ts_diff[0]  # Set the initial value

        # Perform vectorized addition to calculate the integrated series
#        ts[1:] = ts_diff[1:] + ts_base[:-1]

#        return ts.tolist()




D=np.concatenate((tos_index,so_index,ekman_pumping_index,mld_index,b_index,pd_index),axis=0)
D=D.reshape(1980,6)

for j in range(D.shape[1]):
      D[:,j]=ts_diff(D[:,j])
# Use the mask to exclude NaN values
T=np.array(no3_index)

T1=np.zeros((1980,))
T1[:,]=ts_diff(T)
print('D',D.shape)
#if k>0:
#   D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:5],pc1_so[:,:3],pc1_zos[:,:5]),axis=1) # full case

#D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)

#if i>0:
#   #D=np.concatenate((X_no3[:-Z[i],np.newaxis],X_po4[:-Z[i],np.newaxis],pc1_tos[:-Z[i],:10],pc1_so[:-Z[i],:5],pc1_zos[:-Z[i],:5]),axis=1)

X_ss, Y_mm =  split_sequences(D,T1,108,108)
print("X_ss",X_ss.shape)
print("y_mm",Y_mm.shape)
train_ratio=0.7
train_len = round(len(X_ss[:-(108+108+264)]) * train_ratio)
test_len=126 #150/3


threshold = 0.5  # Adjust this threshold as needed for binary classification
#X_train, y_train = create_binary_sequences(train_data, input_length, output_length)
#X_test, y_test = create_binary_sequences(test_data, input_length, output_length)
X_train,y_train=X_ss[:-(108+108+264)],Y_mm[:-(108+108+264)]
X_train = X_train.reshape(X_train.shape[0], -1)

X_test,y_test=X_ss[-(108+108+264):],Y_mm[-(108+108+264):]
X_test = X_test.reshape(X_test.shape[0], -1)

# Create and fit a logistic regression model
alpha = 0.5  # Regularization strength (adjust as needed)
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)

skill_reg=np.zeros((18))
test_len=108+108+264
prediction1=np.zeros((264))
prediction2=np.zeros((264))
prediction3=np.zeros((264))
prediction4=np.zeros((264))
prediction5=np.zeros((264))
prediction6=np.zeros((264))
prediction7=np.zeros((264))
prediction8=np.zeros((264))
prediction9=np.zeros((264))
prediction10=np.zeros((264))
prediction11=np.zeros((264))
prediction12=np.zeros((264))
prediction13=np.zeros((264))
prediction14=np.zeros((264))
prediction15=np.zeros((264))
prediction16=np.zeros((264))
prediction17=np.zeros((264))


for N in range(264):
  if N==0:

     X_test, Y_test= X_ss[-108:],Y_mm[-108:]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           no3_index[-108:,],
           start = no3_index[-108-1,]
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
           no3_index[-108-N:-N,],
           start = no3_index[-108-N-1,]
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
A=np.corrcoef(prediction1[::-1],no3_index[-264:,])
skill_reg[9]=A[1][0]
A=np.corrcoef(prediction2[::-1],no3_index[-264-Q:-12,])
skill_reg[8]=A[1][0]
A=np.corrcoef(prediction3[::-1],no3_index[-264-Q*2:-24,])
skill_reg[7]=A[1][0]
A=np.corrcoef(prediction4[::-1],no3_index[-264-Q*3:-36,])
skill_reg[6]=A[1][0]
A=np.corrcoef(prediction5[::-1],no3_index[-264-Q*4:-48,])
skill_reg[5]=A[1][0]
A=np.corrcoef(prediction6[::-1],no3_index[-264-Q*5:-60,])
skill_reg[4]=A[1][0]
A=np.corrcoef(prediction7[::-1],no3_index[-264-Q*6:-72,])
skill_reg[3]=A[1][0]
A=np.corrcoef(prediction8[::-1],no3_index[-264-Q*7:-84,])
skill_reg[2]=A[1][0]
A=np.corrcoef(prediction9[::-1],no3_index[-264-Q*8:-96,])
skill_reg[1]=A[1][0]
skill_reg[0]=1

plt.figure()
plt.plot(skill_reg,'b')
plt.ylabel("skill",fontsize=13)
plt.xlabel("Time (years)",fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.savefig('skill_NO3_reg.pdf')


from sklearn.linear_model import LinearRegression

model =LinearRegression()#LinearRegression()
alpha = 0.5  # Regularization strength (adjus

threshold = 0.5  # Adjust this threshold as needed for binary classification
#X_train, y_train = create_binary_sequences(train_data, input_length, output_length)
#X_test, y_test = create_binary_sequences(test_data, input_length, output_length)
X_train,y_train=X_ss[:-(108+108+264)],Y_mm[:-(108+108+264)]
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
prediction10=np.zeros((264))
prediction11=np.zeros((264))
prediction12=np.zeros((264))
prediction13=np.zeros((264))
prediction14=np.zeros((264))
prediction15=np.zeros((264))
prediction16=np.zeros((264))
prediction17=np.zeros((264))
prediction18=np.zeros((264))


skill_linreg=np.zeros((18))
for N in range(264):
  if N==0:

     X_test, Y_test= X_ss[-108:],Y_mm[-108:]
     X_test = X_test.reshape(X_test.shape[0], -1)
     y_pred = model.predict(X_test)
     Z=ts_int(
             y_pred[-1,:].tolist(),
           no3_index[-108:,],
           start = no3_index[-108-1,]
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
           no3_index[-108-N:-N,],
           start = no3_index[-108-N-1,]
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
A=np.corrcoef(prediction1[::-1],no3_index[-264:,])
skill_linreg[9]=A[1][0]
A=np.corrcoef(prediction2[::-1],no3_index[-264-Q:-12,])
skill_linreg[8]=A[1][0]
A=np.corrcoef(prediction3[::-1],no3_index[-264-Q*2:-24,])
skill_linreg[7]=A[1][0]
A=np.corrcoef(prediction4[::-1],no3_index[-264-Q*3:-36,])
skill_linreg[6]=A[1][0]
A=np.corrcoef(prediction5[::-1],no3_index[-264-Q*4:-48,])
skill_linreg[5]=A[1][0]
A=np.corrcoef(prediction6[::-1],no3_index[-264-Q*5:-60,])
skill_linreg[4]=A[1][0]
A=np.corrcoef(prediction7[::-1],no3_index[-264-Q*6:-72,])
skill_linreg[3]=A[1][0]
A=np.corrcoef(prediction8[::-1],no3_index[-264-Q*7:-84,])
skill_linreg[2]=A[1][0]
A=np.corrcoef(prediction9[::-1],no3_index[-264-Q*8:-96,])
skill_linreg[1]=A[1][0]
skill_linreg[0]=1

plt.figure()
plt.plot(skill_linreg,'g')
plt.ylabel("skill",fontsize=13)
plt.xlabel("Time (years)",fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.savefig('skill_NO3_linearregression.png')
