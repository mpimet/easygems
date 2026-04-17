import numpy as np
import xarray as xr
from scipy.spatial import ConvexHull, KDTree

from . import transform


def compute_weights_delaunay(points, xi):
    """Compute remapping weights for linear interpolation.

    Uses a convex hull of 3D coordinates as a spherical Delaunay triangulation,
    avoiding all projection artifacts at poles and the dateline.

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
    # Convert coordinates into Cartesian coordinates
    # for triangulation and weight computation.
    src_xyz = transform.latlon2xyz(np.array(points).T)
    tgt_xyz = transform.latlon2xyz(np.stack(xi, axis=-1))

    simplices = ConvexHull(src_xyz).simplices

    centroids = src_xyz[simplices].mean(axis=1)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_tree = KDTree(centroids)

    tgt_xyz /= np.linalg.norm(tgt_xyz, axis=1, keepdims=True)
    _, cidx = centroid_tree.query(tgt_xyz)
    src_idx = simplices[cidx]

    # Compute barycentric weights in 3D
    verts_xyz = src_xyz[src_idx]
    v0, v1, v2 = verts_xyz[:, 0], verts_xyz[:, 1], verts_xyz[:, 2]
    p = tgt_xyz

    area_total = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    w0 = np.linalg.norm(np.cross(v1 - p, v2 - p), axis=1) / area_total
    w1 = np.linalg.norm(np.cross(v2 - p, v0 - p), axis=1) / area_total
    w2 = 1.0 - w0 - w1

    weights = np.stack([w0, w1, w2], axis=-1)

    return xr.Dataset(
        data_vars={
            "src_idx": (("tgt_idx", "tri"), src_idx),
            "weights": (("tgt_idx", "tri"), weights),
            "valid": (("tgt_idx",), np.full(cidx.size, fill_value=True)),
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
