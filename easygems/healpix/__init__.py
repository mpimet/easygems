import warnings

import numpy as np
import cf_xarray as cf_xarray
import xarray as xr
import healpix

from ..resample import HEALPixResampler
from ..show import map_show, map_contour


def get_nest(dx):
    return dx.cf["grid_mapping"].healpix_order in ["nest", "nested"]


def get_nside(dx):
    try:
        return dx.cf["grid_mapping"].healpix_nside
    except (AttributeError, KeyError):
        if dx.squeeze().ndim > 1:
            raise ValueError(
                "Cannot infer the HEALPix resolution from a multidimensional dataset.\n"
                "Consider adding a coordinate reference system to the dataset or passing a one-dimensional array instead.\n"
                "See also: easygems.healpix.attach_coords\n"
                "Reference: https://easy.gems.dkrz.de/Processing/datasets/remapping.html#storing-the-coordinate-reference-system"
            )
        return healpix.npix2nside(dx.size)


def get_npix(dx):
    return healpix.nside2npix(get_nside(dx))


def get_extent_mask(dx, extent):
    lon = dx.lon
    lat = dx.lat

    w, e, s, n = extent  # Shortcut for N/S/E/W bounds

    is_in_lon = (lon - w) % 360 < (e - w) % 360  # consider sign change
    is_in_lat = (lat > s) & (lat < n)

    return is_in_lon & is_in_lat


def get_full_chunks(indices, chunksize):
    """Return indices of complete chunks, given a list of indices and a chunksize."""
    used_chunks = np.unique(np.asarray(indices) // chunksize)

    return (used_chunks[:, np.newaxis] * chunksize + np.arange(chunksize)).flatten()


def isel_extent(dx, extent):
    return np.arange(get_npix(dx))[get_extent_mask(dx, extent)]


def broadcast_array(dx, fill_value=np.nan):
    """Broadcast a limited-area HEALPix array to full shape."""
    arr = np.full_like(dx, fill_value, shape=get_npix(dx))
    arr[dx.cell] = dx.values

    return arr


def fix_crs(ds: xr.Dataset):
    # remove crs dimension (crs should really be 0-dimensional, but sometimes we keep a dimension
    # to be compatible with netcdf
    grid_mapping_var = ds.cf["grid_mapping"].name
    ds = ds.drop_vars(grid_mapping_var).assign_coords(
        {grid_mapping_var: ((), 0, ds.cf["grid_mapping"].attrs)}
    )
    return ds


def guess_crs(ds: xr.Dataset):
    warnings.warn(
        "No CRS coordinate was found. Attempting to infer it from the dataset shape. Please check the result!",
        stacklevel=4,
    )

    if "cell" in ds.dims:
        pix = ds.cell
    elif "values" in ds.dims:
        pix = ds.values
    elif "value" in ds.dims:
        pix = ds.value

    crs = xr.DataArray(
        name="crs",
        attrs={
            "grid_mapping_name": "healpix",
            "healpix_nside": healpix.npix2nside(pix.size),
            "healpix_order": "nest",
        },
    )
    return ds.assign_coords(crs=crs)


def attach_coords(ds: xr.Dataset, signed_lon=False):
    try:
        ds.cf["grid_mapping"]
    except KeyError:
        ds = guess_crs(ds)
    else:
        ds = fix_crs(ds)

    cell = ds.get("cell") if "cell" in ds.dims else np.arange(get_npix(ds))

    lons, lats = healpix.pix2ang(
        get_nside(ds), cell.astype("i8"), nest=get_nest(ds), lonlat=True
    )
    if signed_lon:
        lons = np.where(lons <= 180, lons, lons - 360)
    else:
        # Both healpy and healpix produce longitudes in the range [-45, 360]
        # While this is mathematically valid, it may be unexpected in Earth system science.
        lons %= 360
    return ds.assign_coords(
        cell=cell,
        lat=(
            ("cell",),
            lats,
            {"units": "degree_north", "standard_name": "latitude", "axis": "Y"},
        ),
        lon=(
            ("cell",),
            lons,
            {"units": "degree_east", "standard_name": "longitude", "axis": "X"},
        ),
    )


def healpix_show(var, method="nearest", nest=True, **kwargs):
    r = HEALPixResampler(nside=get_nside(var), nest=nest, method=method)

    return map_show(var, resampler=r, **kwargs)


def healpix_contour(var, method="nearest", nest=True, **kwargs):
    r = HEALPixResampler(nside=get_nside(var), nest=nest, method=method)

    return map_contour(var, resampler=r, **kwargs)


__all__ = [
    "get_nest",
    "get_nside",
    "get_npix",
    "get_extent_mask",
    "get_full_chunks",
    "isel_extent",
    "fix_crs",
    "attach_coords",
    "healpix_show",
    "healpix_contour",
]
