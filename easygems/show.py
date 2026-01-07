import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl import geoaxes


def create_geoaxis(add_coastlines=True, **subplot_kw):
    """Convenience function to create a figure with a default map projection."""
    if "projection" not in subplot_kw:
        subplot_kw["projection"] = ccrs.Robinson(central_longitude=-135.58)

    # The GeoAxes check in get_current_geoaxis() always creates a figure.
    # Here, we need to ensure that we reuse this figure instead of creating a new one.
    fig = plt.gcf()
    ax = fig.add_subplot(111, **subplot_kw)
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


def sample_image(arr, resampler, ax, antialias=False):
    """Sample an image in pixel-space."""
    # Get axis size in pixels
    _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)

    if antialias:
        nx *= 2
        ny *= 2

    # Create 2d-arrays for x/y coordinates in Cartopy coordinates
    xlims, ylims = ax.get_xlim(), ax.get_ylim()

    dx = (xlims[1] - xlims[0]) / nx
    dy = (ylims[1] - ylims[0]) / ny
    xvals, yvals = np.meshgrid(
        np.linspace(xlims[0] + dx / 2, xlims[1] - dx / 2, nx),
        np.linspace(ylims[0] + dy / 2, ylims[1] - dy / 2, ny),
    )

    # Transform to lat/lon coordinates
    latlon = ccrs.PlateCarree().transform_points(
        ax.projection, xvals, yvals, np.zeros_like(xvals)
    )
    valid = np.all(np.isfinite(latlon), axis=-1)

    # Fill valid points in "plot array"
    res = np.full(latlon.shape[:-1], np.nan, dtype=arr.dtype)
    res[valid] = resampler.get_values(arr, latlon[valid, :2])

    return xr.DataArray(res, coords=[("y", yvals[:, 0]), ("x", xvals[0, :])])


def map_show(
    var, resampler, ax=None, antialias=False, dpi=None, add_coastlines=True, **kwargs
):
    """Plot a variable on a Cartopy GeoAxes.

    Parameters:
        var: array-like
            Variable to plot
        resampler: Resampler
            Resampling from lon/lat coordinates to source index
        ax: cartopy.GeoAxis
        antialias: bool
            If True, sample at double the resolution for anti-aliasing
        dpi: int
            Pixel resolution of created figure
        add_coastlines: bool
            Create GeoAxes with coastlines, if none is passed
        **kwargs: Additional keyword argument passed to `plt.imshow()`
    """
    if ax is None:
        ax = get_current_geoaxis(add_coastlines=add_coastlines)

    if dpi is not None:
        ax.get_figure().set_dpi(dpi)

    img = sample_image(var, resampler, ax, antialias)

    # Use the original map extent, as the x and y coordinates in img
    # are shifted by half a pixel to create middle points.
    extent = ax.get_xlim() + ax.get_ylim()

    return ax.imshow(img, extent=extent, origin="lower", **kwargs)


def map_contour(
    var, resampler, ax=None, antialias=False, dpi=None, add_coastlines=True, **kwargs
):
    """Plot a variable on a Cartopy GeoAxes.

    Parameters:
        var: array-like
            Variable to plot
        resampler: Resampler
            Resampling from lon/lat coordinates to source index
        ax: cartopy.GeoAxis
        antialias: bool
            If True, sample at double the resolution for anti-aliasing
        dpi: int
            Pixel resolution of created figure
        add_coastlines: bool
            Create GeoAxes with coastlines, if none is passed
        **kwargs: Additional keyword argument passed to `plt.imshow()`
    """
    if ax is None:
        ax = get_current_geoaxis(add_coastlines=add_coastlines)

    if dpi is not None:
        ax.get_figure().set_dpi(dpi)

    img = sample_image(var, resampler, ax, antialias)

    return ax.contour(img.x, img.y, img, **kwargs)
