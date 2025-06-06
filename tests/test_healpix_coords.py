import pytest
from easygems.healpix import attach_coords, get_nside

import cf_xarray as cf_xarray
import numpy as np
import xarray as xr


@pytest.fixture(params=["crs", "healpix"])
def raw_ds(request):
    crs_name = request.param
    return xr.Dataset(
        coords={
            crs_name: (
                ("crs",),
                [0],
                {
                    "grid_mapping_name": "healpix",
                    "healpix_nside": 1,
                    "healpix_order": "nest",
                },
            )
        }
    )


def test_attach_coords_fixes_crs(raw_ds):
    ds = attach_coords(raw_ds)

    assert ds.cf["grid_mapping"].shape == ()
    assert ds.cf["grid_mapping"].attrs == raw_ds.cf["grid_mapping"].attrs
    assert ds.cf["grid_mapping"].name == raw_ds.cf["grid_mapping"].name


def test_attach_coords_adds_lon_lat(raw_ds):
    ds = attach_coords(raw_ds)

    assert ds.lon.standard_name == "longitude"
    assert ds.lon.axis == "X"

    assert ds.lat.standard_name == "latitude"
    assert ds.lat.axis == "Y"


def test_attach_coords_adds_cell(raw_ds):
    ds = attach_coords(raw_ds)

    assert ds.isel(cell=3).cell == 3


def test_attach_coords_no_crs():
    ds = xr.Dataset(coords={"cell": np.arange(48)})

    ds = attach_coords(ds)

    assert ds.crs
    assert ds.crs.healpix_nside == 2


def test_get_nside(raw_ds):
    assert get_nside(raw_ds) == 1
    assert get_nside(np.arange(12)) == 1
