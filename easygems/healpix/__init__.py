import warnings

import numpy as np
import cf_xarray as cf_xarray
import xarray as xr
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import healpix
from cartopy.mpl import geoaxes
from scipy.interpolate import griddata


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


def healpix_resample(var, xlims, ylims, nx, ny, src_crs, method="nearest", nest=True):
    # NOTE: we want the center coordinate of each pixel, thus we have to
    # compute the linspace over half a pixel size less than the plot's limits
    dx = (xlims[1] - xlims[0]) / nx
    dy = (ylims[1] - ylims[0]) / ny
    xvals = np.linspace(xlims[0] + dx / 2, xlims[1] - dx / 2, nx)
    yvals = np.linspace(ylims[0] + dy / 2, ylims[1] - dy / 2, ny)
    xvals2, yvals2 = np.meshgrid(xvals, yvals)
    latlon = ccrs.PlateCarree().transform_points(
        src_crs, xvals2, yvals2, np.zeros_like(xvals2)
    )
    valid = np.all(np.isfinite(latlon), axis=-1)
    points = latlon[valid].T

    res = np.full(latlon.shape[:-1], np.nan, dtype=var.dtype)

    if method == "nearest":
        pix = healpix.ang2pix(
            get_nside(var),
            theta=points[0],
            phi=points[1],
            nest=nest,
            lonlat=True,
        )
        if var.size < get_npix(var):
            if not isinstance(var, xr.DataArray):
                raise ValueError(
                    "Sparse HEALPix grids are only supported as xr.DataArray"
                )

            # For xr.DataArrays, selection from sparse HEALPix grids is supported
            res[valid] = var.sel(cell=pix, method="nearest").where(
                lambda x: x.cell == pix
            )
        else:
            res[valid] = var[pix]
    elif method == "linear":
        lons, lats = healpix.pix2ang(
            nside=get_nside(var),
            ipix=np.arange(len(var)),
            nest=nest,
            lonlat=True,
        )
        lons = (lons + 180) % 360 - 180

        valid_src = ((lons > points[0].min()) & (lons < points[0].max())) | (
            (lats > points[1].min()) & (lats < points[1].max())
        )

        res[valid] = griddata(
            points=np.asarray([lons[valid_src], lats[valid_src]]).T,
            values=var[valid_src],
            xi=(points[0], points[1]),
            method="linear",
            fill_value=np.nan,
            rescale=True,
        )
    else:
        raise ValueError(f"interpolation method '{method}' not known")

    return xr.DataArray(res, coords=[("y", yvals), ("x", xvals)])


def create_geoaxis(add_coastlines=True, **subplot_kw):
    """Convenience function to create a figure with a default map projection."""
    if "projection" not in subplot_kw:
        subplot_kw["projection"] = ccrs.Robinson(central_longitude=-135.58)

    _, ax = plt.subplots(subplot_kw=subplot_kw)
    ax.set_global()

    if add_coastlines:
        ax.coastlines(color="#333333", linewidth=plt.rcParams["grid.linewidth"])

    return ax


def get_current_geoaxis(**kwargs):
    """Return current axis, if it is a GeoAxes, otherwise create a new one."""
    # `plt.gcf().axes` only checks existing axes, while `plt.gca()` also creates one
    if (ax := plt.gcf().axes) and isinstance(ax[0], geoaxes.GeoAxes):
        return ax[0]
    else:
        return create_geoaxis(**kwargs)


def healpix_show(
    var,
    dpi=None,
    ax=None,
    method="nearest",
    nest=True,
    add_coastlines=True,
    antialias=False,
    **kwargs,
):
    if ax is None:
        ax = get_current_geoaxis(add_coastlines=add_coastlines)
    fig = ax.get_figure()

    if dpi is not None:
        fig.set_dpi(dpi)

    _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)

    if antialias:
        nx *= 2
        ny *= 2

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    im = healpix_resample(var, xlims, ylims, nx, ny, ax.projection, method, nest)

    return ax.imshow(im, extent=xlims + ylims, origin="lower", **kwargs)


def healpix_contour(
    var, dpi=None, ax=None, method="linear", nest=True, add_coastlines=True, **kwargs
):
    if ax is None:
        ax = get_current_geoaxis(add_coastlines=add_coastlines)
    fig = ax.get_figure()

    if dpi is not None:
        fig.set_dpi(dpi)

    _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    im = healpix_resample(var, xlims, ylims, nx, ny, ax.projection, method, nest)

    return ax.contour(im.x, im.y, im, **kwargs)


__all__ = [
    "get_nest",
    "get_nside",
    "get_npix",
    "get_extent_mask",
    "get_full_chunks",
    "isel_extent",
    "fix_crs",
    "attach_coords",
    "healpix_resample",
    "create_geoaxis",
    "get_current_geoaxis",
    "healpix_show",
    "healpix_contour",
]
