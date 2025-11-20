import numpy as np
import xarray as xr

from . import resample


def compute_weights_delaunay(points, xi):
    """Compute remapping weights for linear interpolation.

    The interpolation is based on a Delaunay triangulation on the sphere [0].

    [0]: https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/

    Args:
        points (tuple[ndarrays]): Tuple with source grid coordinates.
        xi (tuple[ndarrays]): Tuple with target grid coordinates.

    Returns:
        xr.Dataset: Remapping information
            "src_idx": Array with source grid indices
            "weights": Remapping weights
            "valid": Boolean mask determining if a target point is within the
                triangulation. If it's outside, the target point will not
                receive any data after interpolation any user might want
                to set it to some missing value.

    See also:
        `apply_weights`
    """
    resampler = resample.DelaunayResampler(points[0], points[1])
    src_idx, weights, valid = resampler.get_weights(np.stack(xi, axis=-1))

    return xr.Dataset(
        data_vars={
            "src_idx": (("tgt_idx", "tri"), src_idx),
            "weights": (("tgt_idx", "tri"), weights),
            "valid": (("tgt_idx",), valid),
        }
    )


def apply_weights(var, src_idx, weights, valid):
    """Apply given remapping weights.

    Args:
        var (ndarray): Array to remap.
        kwargs: Remapping weights as returned by `compute_weights`.

    Returns:
        ndarray: Remapped values

    See also:
        `compute_weights`
    """
    return np.where(valid, (var[src_idx] * weights).sum(axis=-1), np.nan)
