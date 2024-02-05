import numpy as np
import xarray as xr
from scipy.spatial import Delaunay


def compute_weights_delaunay(points, xi):
    """Compute remapping weights for linear interpolation.

    The inerpolation is based on an internally computed Delaunay triangulation.

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
    tri = Delaunay(np.stack(points, axis=-1))  # Compute the triangulation
    targets = np.stack(xi, axis=-1)
    triangles = tri.find_simplex(targets)

    X = tri.transform[triangles, :2]
    Y = targets - tri.transform[triangles, 2]
    b = np.einsum("...jk,...k->...j", X, Y)
    weights = np.concatenate([b, 1 - b.sum(axis=-1)[..., np.newaxis]], axis=-1)
    src_idx = tri.simplices[triangles]
    valid = triangles >= 0

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
