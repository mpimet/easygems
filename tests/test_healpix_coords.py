import pytest
import xarray as xr
from easygems.healpix import attach_coords


@pytest.fixture
def raw_ds():
    return xr.Dataset(
        coords={
            "crs": (
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

    assert ds.crs.shape == ()
    assert ds.crs.attrs == raw_ds.crs.attrs


def test_attach_coords_adds_lon_lat(raw_ds):
    ds = attach_coords(raw_ds)

    assert ds.lon.standard_name == "longitude"
    assert ds.lon.axis == "X"

    assert ds.lat.standard_name == "latitude"
    assert ds.lat.axis == "Y"
