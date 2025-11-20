import healpix as hp
import numpy as np
import xarray as xr
from scipy.spatial import Delaunay, KDTree

from . import transform


class Resampler:
    def get_values(self, m, coords):
        """Return value(s) from a map.

        Parameters:
            m: ndarray, shape (...)
            coords: ndarray, shape (N, 2)

        Returns:
        """
        raise NotImplementedError


class WeightBasedResampler(Resampler):
    def get_weights(self, coords):
        """Return resampling weights for remapping to given coords.

        Parameters:
            coords: ndarray, shape (N, 2)

        Returns:
            src_index: ndarray, shape (N, M)
            weights: ndarray, shape (N, M)
            valid: ndarray, shape (N)
        """
        raise NotImplementedError

    def get_values(self, m, coords):
        coords = np.asarray(coords)
        m = np.asarray(m)

        idx, weights, valid = self.get_weights(coords)
        return np.where(valid, (m[idx] * weights).sum(axis=-1), np.nan)


class IndexBasedResampler(WeightBasedResampler):
    def get_source_indices(self, coords):
        """Return resampling weights for remapping to given coords.

        Parameters:
            coords: ndarray, shape (N, 2)

        Returns:
            src_index: ndarray, shape (N, M)
        """
        raise NotImplementedError

    def get_weights(self, coords):
        idx = self.get_source_indices(coords)[..., np.newaxis]
        weights = np.ones(idx.shape)
        valid = np.ones(idx.shape[:-1], dtype="bool")
        return idx, weights, valid

    def get_values(self, m, coords):
        coords = np.asarray(coords)
        idx = self.get_source_indices(coords)

        if isinstance(m, xr.DataArray):
            # Using coordinate **values** allows sparse HEALPix grids to be supported.
            dim = m.dims[0]
            return m.sel({dim: idx}, method="nearest").where(lambda x: x[dim] == idx)
        else:
            m = np.asarray(m)
            return m[idx]


class NearestHEALPixResampler(IndexBasedResampler):
    def __init__(self, nside, nest=True):
        self.nside = nside
        self.nest = nest

    def get_source_indices(self, coords):
        coords = np.asarray(coords)

        return hp.ang2pix(
            nside=self.nside,
            theta=coords[:, 0],
            phi=coords[:, 1],
            nest=self.nest,
            lonlat=True,
        )


class LinearHEALPixResampler(WeightBasedResampler):
    def __init__(self, nside, nest=True):
        self.nside = nside
        self.nest = nest
        try:
            import healpy
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Install the `healpy` package to enable linear HEALPix interpolation."
            )
        self._healpy = healpy

    def get_weights(self, coords):
        ## TODO
        # if hp.nside2npix(self.nside) != m.size:
        #    raise ValueError(
        #        "Linear interpolation is only supported for full HEALPix maps!"
        #    )

        coords = np.asarray(coords)

        idx, weights = self._healpy.get_interp_weights(
            nside=self.nside,
            theta=coords[:, 0],
            phi=coords[:, 1],
            nest=self.nest,
            lonlat=True,
        )

        idx = idx.T
        weights = weights.T
        valid = np.ones(weights.shape[:-1], dtype="bool")

        return idx, weights, valid


def HEALPixResampler(nside, nest=True, method="nearest"):
    if method == "nearest":
        return NearestHEALPixResampler(nside, nest)
    elif method == "linear":
        return LinearHEALPixResampler(nside, nest)
    else:
        raise ValueError(f"Unsupported interpolation method {method}.")


class KDTreeResampler(IndexBasedResampler):
    """Build a KDTree for lat/lon pairs to find the nearest neighbour."""

    def __init__(self, lon, lat):
        xyz = transform.latlon2xyz(np.array([lon, lat]).T)

        self.tree = KDTree(xyz)

    def get_source_indices(self, coords):
        xyz = transform.latlon2xyz(np.asarray(coords))

        _, idx = self.tree.query(xyz)

        return idx


class DelaunayResampler(WeightBasedResampler):
    """Perform a Delaunay triangulation to find the neighbouring cells.

    References:
        https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/
    """

    def __init__(self, lon, lat):
        xy = transform.latlon2stereographic(np.array([lon, lat]).T)

        self.xyz = transform.latlon2xyz(np.array([lon, lat]).T)
        self.tri = Delaunay(xy)

    def get_weights(self, coords):
        coords = np.asarray(coords)

        # Triangulation in stereographic projection
        xy = transform.latlon2stereographic(coords)
        triangles = self.tri.find_simplex(xy)
        valid = triangles >= 0

        idx = self.tri.simplices[triangles]

        # Compute barycentric weights in 3D
        verts_xyz = self.xyz[idx]
        v0, v1, v2 = verts_xyz[:, 0], verts_xyz[:, 1], verts_xyz[:, 2]
        p = transform.latlon2xyz(coords)

        area_total = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        w0 = np.linalg.norm(np.cross(v1 - p, v2 - p), axis=1) / area_total
        w1 = np.linalg.norm(np.cross(v2 - p, v0 - p), axis=1) / area_total
        w2 = 1.0 - w0 - w1

        weights = np.stack([w0, w1, w2], axis=-1)

        return idx, weights, valid
