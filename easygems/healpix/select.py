import healpix
import numpy as np
import pyproj
import regionmask

from .inspection import get_index, get_nside, is_nested


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


def get_index_regionmask(dx, region, defined_regions=None):
    """Return the HEALPix indices inside a defined region.

    Note:
        Any defined region following the `regionmask` specification can be used.
        The AR6 SREX region(s) are the default.

    Reference:
        https://regionmask.readthedocs.io/en/stable/defined_scientific.html
    """
    if "lon" not in dx.variables and "lat" not in dx.variables:
        raise AttributeError("Could not find 'lat' and 'lon' variables.")

    if defined_regions is None:
        defined_regions = regionmask.defined_regions.ar6.all

    is_in_region = defined_regions.mask(dx.lon, dx.lat).isin(
        defined_regions.map_keys(region)
    )

    return get_index(dx)[is_in_region]


def select_regionmask(dx, region, defined_regions=None):
    """Return a subselected HEALPix dataset inside a given AR6 SREX region(s).

    Note:
        Any defined region following the `regionmask` specification can be used.
        The AR6 SREX region(s) are the default.

    Reference:
        https://regionmask.readthedocs.io/en/stable/defined_scientific.html
    """
    idx = get_index_regionmask(dx, region, defined_regions=defined_regions)

    return dx.sel({idx.name: idx})


__all__ = [
    "get_index_extent",
    "select_extent",
    "get_index_section",
    "select_section",
    "get_index_regionmask",
    "select_regionmask",
]
