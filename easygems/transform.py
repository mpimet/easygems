import numpy as np


def latlon2xyz(coords, R=1.0):
    """Convert geocentric coordaintes into 3D Cartesian."""
    lon = np.deg2rad(coords[:, 0])
    lat = np.deg2rad(coords[:, 1])

    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)

    return np.stack([x, y, z]).T


def latlon2stereographic(coords, R=1.0):
    """Stereographic projection centered on the North Pole."""
    lon = np.deg2rad(coords[:, 0])
    lat = np.deg2rad(coords[:, 1])

    k = 2 * R / (1 + np.sin(lat))

    x = k * np.cos(lat) * np.sin(lon)
    y = k * np.cos(lat) * np.cos(lon)

    return np.stack([x, y]).T
