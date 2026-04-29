from .inspection import get_nside
from ..resample import HEALPixResampler
from ..show import map_show, map_contour


def healpix_show(var, method="nearest", nest=True, **kwargs):
    r = HEALPixResampler(nside=get_nside(var), nest=nest, method=method)

    return map_show(var, resampler=r, **kwargs)


def healpix_contour(var, method="nearest", nest=True, **kwargs):
    r = HEALPixResampler(nside=get_nside(var), nest=nest, method=method)

    return map_contour(var, resampler=r, **kwargs)


__all__ = [
    "healpix_show",
    "healpix_contour",
]
