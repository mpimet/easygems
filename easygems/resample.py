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
            val: ndarray, shape (N)
        """
        raise NotImplementedError


class HEALPixResampler(Resampler):
    """Find the nearest cell on an HEALPix grid."""

    def __init__(self, nside, nest=True, method="nearest"):
        self.nside = nside
        self.nest = nest
        self.method = method

    def get_interp_values(self, m, coords):
        try:
            import healpy
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Install the `healpy` package to enable linear HEALPix interpolation."
            )

        if hp.nside2npix(self.nside) != m.size:
            raise ValueError(
                "Linear interpolation is only supported for full HEALPix maps!"
            )

        coords = np.asarray(coords)

        return healpy.get_interp_val(
            m,
            theta=coords[:, 0],
            phi=coords[:, 1],
            nest=self.nest,
            lonlat=True,
        )

    def get_nearest_values(self, m, coords):
        coords = np.asarray(coords)

        idx = hp.ang2pix(
            nside=self.nside,
            theta=coords[:, 0],
            phi=coords[:, 1],
            nest=self.nest,
            lonlat=True,
        )

        if m.size < hp.nside2npix(self.nside):
            if isinstance(m, xr.DataArray):
                # Using coordinate **values** allows sparse HEALPix grids to be supported.
                dim = m.dims[0]
                if dim in m.coords:
                    return m.sel({dim: idx}, method="nearest").where(
                        lambda x: x[dim] == idx
                    )
            raise ValueError(
                "Sparse HEALPix grids need to be passed "
                "as an xr.DataArray that contains the pixel coordinates."
            )
        else:
            return m[idx]

    def get_values(self, m, coords):
        if self.method == "nearest":
            return self.get_nearest_values(m, coords)
        elif self.method == "linear":
            return self.get_interp_values(m, coords)
        else:
            raise ValueError(f"Unsupported interpolation method {self.method}.")


class KDTreeResampler(Resampler):
    """Build a KDTree for lat/lon pairs to find the nearest neighbour."""

    def __init__(self, lon, lat):
        xyz = transform.latlon2xyz(np.array([lon, lat]).T)

        self.tree = KDTree(xyz)

    def get_values(self, m, coords):
        xyz = transform.latlon2xyz(np.asarray(coords))

        _, idx = self.tree.query(xyz)

        return np.asarray(m)[idx]


class DelaunayResampler(Resampler):
    """Perform a Delaunay triangulation to find the neighbouring cells.

    References:
        https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/
    """

    def __init__(self, lon, lat):
        xy = transform.latlon2stereographic(np.array([lon, lat]).T)

        self.xyz = transform.latlon2xyz(np.array([lon, lat]).T)
        self.tri = Delaunay(xy)

    def get_values(self, m, coords):
        coords = np.asarray(coords)
        m = np.asarray(m)

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

        return np.where(valid, (m[idx] * weights).sum(axis=-1), np.nan)
