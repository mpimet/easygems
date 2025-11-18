import healpix as hp
import numpy as np
import xarray as xr
from scipy.spatial import Delaunay, KDTree


def latlon2xyz(coords, R=1.0):
    """Convert geographic coordaintes into 3D Cartesian."""
    lon = np.deg2rad(coords[:, 0])
    lat = np.deg2rad(coords[:, 1])

    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)

    return np.stack([x, y, z]).T


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

        if isinstance(m, xr.DataArray):
            # Using coordinate **values** allows sparse HEALPix grids to be supported.
            dim = m.dims[0]
            return m.sel({dim: idx}, method="nearest").where(lambda x: x[dim] == idx)
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
        xyz = latlon2xyz(np.array([lon, lat]).T)

        self.tree = KDTree(xyz)

    def get_values(self, m, coords):
        xyz = latlon2xyz(np.asarray(coords))

        _, idx = self.tree.query(xyz)

        return np.asarray(m)[idx]


class DelaunayResampler(Resampler):
    """Perform a Delaunay triangulation to find the neighbouring cells."""

    def __init__(self, lon, lat):
        lon = (np.asarray(lon) + 180) % 360 - 180  # Ensure lon [-180, 180]
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
