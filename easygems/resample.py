import healpix as hp
import numpy as np
import xarray as xr
from scipy.spatial import Delaunay, KDTree


class Resampler:
    def get_values(self, m, coords):
        """Return value(s) from a map.

        Parameters:
            m: ndarray, shape (...)
            coords: ndarray, shape (N, 2)

        Returns:
            val: ndarray, shape (N)
        """
        raise NotImplementedError


class HEALPixResampler(Resampler):
    """Find the nearest cell on an HEALPix grid."""

    def __init__(self, nside, nest=True):
        self.nside = nside
        self.nest = nest

    def get_values(self, m, coords):
        coords = np.asarray(coords)

        idx = hp.ang2pix(
            nside=self.nside,
            theta=coords[:, 0],
            phi=coords[:, 1],
            nest=self.nest,
            lonlat=True,
        )

        if isinstance(m, xr.DataArray):
            # Using coordinate **values** allows sparse HEALPix grids to be supported.
            dim = m.dims[0]
            return m.sel({dim: idx}, method="nearest").where(lambda x: x[dim] == idx)
        else:
            return m[idx]


class KDTreeResampler(Resampler):
    """Build a KDTree for lat/lon pairs to find the nearest neighbour."""

    def __init__(self, lon, lat):
        self.tree = KDTree(np.array([lon, lat]).T)

    def get_values(self, m, coords):
        _, idx = self.tree.query(coords)

        return np.asarray(m)[idx]


class DelaunayResampler(Resampler):
    """Perform a Delaunay triangulation to find the neighbouring cells."""

    def __init__(self, lon, lat):
        self.tri = Delaunay(np.stack([lon, lat], axis=-1))

    def get_values(self, m, coords):
        triangles = self.tri.find_simplex(coords)

        X = self.tri.transform[triangles, :2]
        Y = coords - self.tri.transform[triangles, 2]

        idx = self.tri.simplices[triangles]

        b = np.einsum("...jk,...k->...j", X, Y)
        weights = np.concatenate([b, 1 - b.sum(axis=-1)[..., np.newaxis]], axis=-1)
        valid = triangles >= 0

        return np.where(valid, (np.asarray(m)[idx] * weights).sum(axis=-1), np.nan)
