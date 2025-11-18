from easygems import resample

import numpy as np
import xarray as xr


def test_delaunay_resampler():
    lon = [0, 90, 45]
    lat = [0, 0, 90]
    val = [1, 2, 3]

    r = resample.DelaunayResampler(lon=lon, lat=lat)

    assert r.get_values(val, [[45, 45]]) == np.array([2.25])


def test_healpix_resampler():
    nside = 1
    nest = True
    val = np.arange(12)

    r = resample.HEALPixResampler(nside=nside, nest=nest)

    assert r.get_values(val, [[0, 0]]) == np.array([4])


def test_healpix_resampler_sparse():
    nside = 1
    nest = True
    val = xr.DataArray([3, 4, 5], coords={"cell": [3, 4, 5]})

    r = resample.HEALPixResampler(nside=nside, nest=nest)

    assert r.get_values(val, [[0, 0]]) == np.array([4])


def test_kdtree_resampler():
    lon = [-180, 0, 180]
    lat = [-90, 0, 90]
    val = [1, 2, 3]

    r = resample.KDTreeResampler(lon=lon, lat=lat)

    res = r.get_values(val, [[0, 0]])
    assert np.array_equal(res, np.array([2]))

    res = r.get_values(val, [[0, 0], [-170, -90]])
    assert np.array_equal(res, np.array([2, 1]))


def test_kdtree_resampler_lon_range():
    lon = [10, 190, 350]
    lat = [0, 0, 0]
    val = [1, 2, 3]

    r = resample.KDTreeResampler(lon=lon, lat=lat)

    res = r.get_values(val, [[10, 0], [-170, 0], [-10, 0]])
    assert np.array_equal(res, np.array([1, 2, 3]))
