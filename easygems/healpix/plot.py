from functools import wraps

from .inspection import get_nside
from ..resample import HEALPixResampler
from ..show import map_show, map_contour, map_contourf


@wraps(map_show, assigned=("__doc__",))
def healpix_show(var, method="nearest", nest=True, **kwargs):
    r = HEALPixResampler(nside=get_nside(var), nest=nest, method=method)

    return map_show(var, resampler=r, **kwargs)


@wraps(map_contour, assigned=("__doc__",))
def healpix_contour(var, method="nearest", nest=True, **kwargs):
    r = HEALPixResampler(nside=get_nside(var), nest=nest, method=method)

    return map_contour(var, resampler=r, **kwargs)


@wraps(map_contourf, assigned=("__doc__",))
def healpix_contourf(var, method="nearest", nest=True, **kwargs):
    r = HEALPixResampler(nside=get_nside(var), nest=nest, method=method)

    return map_contourf(var, resampler=r, **kwargs)


__all__ = [
    "healpix_show",
    "healpix_contour",
    "healpix_contourf",
]
