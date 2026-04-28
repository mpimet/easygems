import warnings

import numpy as np
import cf_xarray as cf_xarray
import xarray as xr
import healpix
import pyproj

from ..resample import HEALPixResampler
from ..show import map_show, map_contour


def get_nest(dx):
    warnings.warn(
        "This function is deprecated. Use `is_nested()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_nested(dx)


def is_nested(dx):
    try:
        # Check HEALPix grid parameters compliant with CF Conventions
        indexing_scheme = dx.cf["grid_mapping"].indexing_scheme

        valid_parameters = ("nested", "ring", "nuniq", "zuniq")
        if indexing_scheme not in valid_parameters:
            raise ValueError(
                f"Indexing scheme '{indexing_scheme}' is not in the list of valid parameters: {valid_parameters}\n"
                "Further details: https://cfconventions.org/cf-conventions/cf-conventions.html#healpix"
            )

        return indexing_scheme == "nested"
    except AttributeError:
        # Check legacy HEALPix grid parameters
        indexing_scheme = dx.cf["grid_mapping"].healpix_order

        return indexing_scheme in ["nest", "nested"]


def get_nside(dx):
    try:
        grid_mapping = dx.cf["grid_mapping"]
    except (KeyError, AttributeError):
        # Catch no CF (KeyError) and no grid mapping (AttributeError)
        if dx.squeeze().ndim > 1:
            raise ValueError(
                "Cannot infer the HEALPix resolution from a multidimensional dataset.\n"
                "Consider adding a coordinate reference system to the dataset or passing a one-dimensional array instead.\n"
                "See also: easygems.healpix.attach_coords\n"
                "Reference: https://easy.gems.dkrz.de/Processing/datasets/remapping.html#storing-the-coordinate-reference-system"
            )
        return healpix.npix2nside(dx.size)
    else:
        try:
            return healpix.order2nside(grid_mapping.refinement_level)
        except AttributeError:
            return grid_mapping.healpix_nside


def get_npix(dx):
    return healpix.nside2npix(get_nside(dx))


def get_index_name(dx):
    """Return the name of the (most likely) HEALPix index."""
    for name, coord in dx.coords.items():
        if coord.attrs.get("standard_name") == "healpix_index":
            return name

    for possible_index in ("cell", "value", "values"):
        if possible_index in dx.dims:
            return possible_index

    raise KeyError("Could not find HEALPix index.")


def get_index(dx, dtype=np.int64):
    """Return the (most likely) HEALPix index."""
    healpix_index = get_index_name(dx)
    return xr.DataArray(
        name=healpix_index,
        dims=(healpix_index,),
        data=dx[healpix_index].values.astype(dtype),
        attrs={"standard_name": "healpix_index"},
    )


def get_index_extent(dx, extent):
    """Return indices of all HEALPix cells within a geospatial extent."""
    healpix_index = get_index(dx)

    # Get coordinates and N/S/E/W bounds
    lon = dx.lon
    lat = dx.lat
    w, e, s, n = extent

    # Compute boolean mask
    is_in_lon = (lon - w) % 360 < (e - w) % 360  # consider sign change
    is_in_lat = (lat > s) & (lat < n)
    is_in_extent = is_in_lon & is_in_lat

    return healpix_index.where(is_in_extent.compute(), drop=True).astype(
        healpix_index.dtype
    )


def select_extent(dx, extent):
    """Return a subselected HEALPix dataset within a geospatial extent."""
    idx = get_index_extent(dx, extent)

    return dx.sel({idx.name: idx})


def get_index_section(dx, lon1, lat1, lon2, lat2, nsample=3600):
    """Return indices of all HEALPix cells along a geospatial section."""
    healpix_index = get_index(dx)

    coords = np.array(pyproj.Geod(ellps="WGS84").npts(lon1, lat1, lon2, lat2, nsample))
    section_indices = healpix.ang2pix(
        get_nside(dx),
        coords[:, 0],
        coords[:, 1],
        nest=is_nested(dx),
        lonlat=True,
    )
    section_indices = np.unique(section_indices)

    is_on_section = healpix_index.isin(section_indices)

    return healpix_index.where(is_on_section.compute(), drop=True).astype(
        healpix_index.dtype
    )


def select_section(dx, lon1, lat1, lon2, lat2, **kwargs):
    """Return indices of all HEALPix cells along a geospatial section."""
    idx = get_index_section(dx, lon1, lat1, lon2, lat2, **kwargs)

    return dx.sel({idx.name: idx})


def get_full_chunks(indices, chunksize):
    """Return indices of complete chunks, given a list of indices and a chunksize."""
    used_chunks = np.unique(np.asarray(indices) // chunksize)

    return (used_chunks[:, np.newaxis] * chunksize + np.arange(chunksize)).flatten()


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


def healpix_show(var, method="nearest", nest=True, **kwargs):
    r = HEALPixResampler(nside=get_nside(var), nest=nest, method=method)

    return map_show(var, resampler=r, **kwargs)


def healpix_contour(var, method="nearest", nest=True, **kwargs):
    r = HEALPixResampler(nside=get_nside(var), nest=nest, method=method)

    return map_contour(var, resampler=r, **kwargs)


__all__ = [
    "is_nested",
    "get_index",
    "get_index_name",
    "get_index_extent",
    "select_extent",
    "get_index_section",
    "select_section",
    "get_nside",
    "get_npix",
    "get_full_chunks",
    "fix_crs",
    "attach_coords",
    "healpix_show",
    "healpix_contour",
]
