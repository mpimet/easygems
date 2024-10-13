import pytest
from easygems.healpix import attach_coords

import cf_xarray as cf_xarray
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


def test_attach_coords_adds_lon_lat(raw_ds):
    ds = attach_coords(raw_ds)

    assert ds.lon.standard_name == "longitude"
    assert ds.lon.axis == "X"

    assert ds.lat.standard_name == "latitude"
    assert ds.lat.axis == "Y"


def test_attach_coords_adds_cell(raw_ds):
    ds = attach_coords(raw_ds)

    assert ds.isel(cell=3).cell == 3
