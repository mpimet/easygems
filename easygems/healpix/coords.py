import warnings

import numpy as np
import cf_xarray as cf_xarray
import xarray as xr
import healpix

from .inspection import get_index, get_npix, get_nside, is_nested


def fix_crs(ds: xr.Dataset):
    # remove crs dimension (crs should really be 0-dimensional, but sometimes we keep a dimension
    # to be compatible with netcdf
    grid_mapping_var = ds.cf["grid_mapping"].name
    ds = ds.drop_vars(grid_mapping_var).assign_coords(
        {
            grid_mapping_var: (
                (),
                0,
                {
                    # Use CF compliant HEALPix map parameters.
                    # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.13/cf-conventions.html#healpix
                    "grid_mapping_name": "healpix",
                    "refinement_level": healpix.nside2order(get_nside(ds)),
                    "indexing_scheme": "nested" if is_nested(ds) else "ring",
                },
            )
        }
    )
    return ds


def guess_crs(ds: xr.Dataset):
    warnings.warn(
        "No CRS coordinate was found. Attempting to infer it from the dataset shape. Please check the result!",
        stacklevel=4,
    )

    pix = get_index(ds)

    crs = xr.DataArray(
        name="crs",
        attrs={
            "grid_mapping_name": "healpix",
            "refinement_level": healpix.nside2order(healpix.npix2nside(pix.size)),
            "indexing_scheme": "nested",
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

    cell = ds.get("cell").values if "cell" in ds.dims else np.arange(get_npix(ds))

    lons, lats = healpix.pix2ang(
        get_nside(ds), cell.astype("i8"), nest=is_nested(ds), lonlat=True
    )
    if signed_lon:
        lons = np.where(lons <= 180, lons, lons - 360)
    else:
        # Both healpy and healpix produce longitudes in the range [-45, 360]
        # While this is mathematically valid, it may be unexpected in Earth system science.
        lons %= 360
    return ds.assign_coords(
        cell=(
            ("cell",),
            cell,
            {"standard_name": "healpix_index"},
        ),
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


__all__ = [
    "fix_crs",
    "guess_crs",
    "attach_coords",
]
