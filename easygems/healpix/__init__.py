import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import healpy


def get_nest(dx):
    return dx.crs.healpix_order == "nest"


def get_nside(dx):
    return dx.crs.healpix_nside


def get_npix(dx):
    return healpy.nside2npix(dx.crs.healpix_nside)


def isel_extent(dx, extent):
    lon = dx.longitude
    lat = dx.latitude

    w, e, s, n = extent  # Shortcut for N/S/E/W bounds

    is_in_lon = (lon - w) % 360 < (e - w) % 360  # consider sign change
    is_in_lat = (lat > s) & (lat < n)

    return np.arange(get_npix(dx))[is_in_lon & is_in_lat]


def attach_coords(ds: xr.Dataset, signed_lon=False):
    lons, lats = healpy.pix2ang(
        get_nside(ds), np.arange(ds.dims["cell"]), nest=get_nest(ds), lonlat=True
    )
    if signed_lon:
        lons = np.where(lons <= 180, lons, lons - 360)
    return ds.assign_coords(
        lat=(("cell",), lats, {"units": "degree_north"}),
        lon=(("cell",), lons, {"units": "degree_east"}),
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
        pix = healpy.ang2pix(
            healpy.npix2nside(len(var)),
            theta=points[0],
            phi=points[1],
            nest=nest,
            lonlat=True,
        )
        res[valid] = var[pix]
    elif method == "linear":
        res[valid] = healpy.get_interp_val(
            var, theta=points[0], phi=points[1], nest=True, lonlat=True
        )
    else:
        raise ValueError(f"interpolation method '{method}' not known")

    return xr.DataArray(res, coords=[("y", yvals), ("x", xvals)])


def healpix_show(var, dpi=None, ax=None, method="nearest", nest=True, **kwargs):
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    if dpi is not None:
        fig.set_dpi(dpi)

    _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    im = healpix_resample(var, xlims, ylims, nx, ny, ax.projection, method, nest)

    return ax.imshow(im, extent=xlims + ylims, origin="lower", **kwargs)


def healpix_contour(var, dpi=None, ax=None, method="linear", nest=True, **kwargs):
    if ax is None:
        ax = plt.gca()
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
    "attach_coords",
    "healpix_resample",
    "healpix_show",
    "healpix_contour",
]
