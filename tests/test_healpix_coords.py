import xarray as xr
from easygems.healpix import attach_coords


def test_attach_coords_fixes_crs():
    ds = xr.Dataset(
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

    ds2 = attach_coords(ds)

    assert ds2.crs.shape == ()
    assert ds2.crs.attrs == ds.crs.attrs
