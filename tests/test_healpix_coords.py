from itertools import product

import pytest
from easygems.healpix import attach_coords, get_index, get_nest, get_nside

import cf_xarray as cf_xarray
import numpy as np
import xarray as xr


@pytest.fixture(params=product(["crs", "healpix"], ["legacy", "cf"]))
def raw_ds(request):
    crs_name = request.param[0]

    if request.param[1] == "cf":
        map_parameters = {
            "refinement_level": 0,
            "indexing_scheme": "nested",
        }
    elif request.param[1] == "legacy":
        map_parameters = {
            "healpix_nside": 1,
            "healpix_order": "nest",
        }

    return xr.Dataset(
        coords={
            crs_name: xr.DataArray(
                name=crs_name,
                attrs={
                    "grid_mapping_name": "healpix",
                    **map_parameters,
                },
            )
        }
    )


def test_attach_coords_fixes_crs(raw_ds):
    ds = attach_coords(raw_ds)

    assert ds.cf["grid_mapping"].shape == ()
    assert ds.cf["grid_mapping"].refinement_level == 0
    assert ds.cf["grid_mapping"].indexing_scheme == "nested"
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
    assert ds.crs.refinement_level == 1


def test_get_nside(raw_ds):
    assert get_nside(raw_ds) == 1
    assert get_nside(np.arange(12)) == 1


def test_get_nest(raw_ds):
    assert get_nest(raw_ds)


@pytest.mark.parametrize("known_name", ["cell", "value", "values"])
def test_get_index_byname(raw_ds, known_name):
    """Test if we can find the HEALPix index by looking for known short names."""
    ds = raw_ds.assign_coords({known_name: np.arange(12)})

    assert np.array_equal(get_index(ds), ds[known_name].values)


def test_get_index_bycf(raw_ds):
    """Test if we can find the HEALPix index using CF Conventions."""
    ds = raw_ds.assign_coords(
        {
            "unknown_name": (
                ("unknown_name",),
                np.arange(12),
                {"standard_name": "healpix_index"},
            )
        }
    )

    assert np.array_equal(get_index(ds), ds["unknown_name"].values)
