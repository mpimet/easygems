import warnings

import cf_xarray as cf_xarray
import healpix
import numpy as np
import xarray as xr


def get_nest(dx):
    warnings.warn(
        "This function is deprecated. Use `is_nested()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_nested(dx)


def is_nested(dx):
    try:
        # Check HEALPix grid parameters compliant with CF Conventions
        indexing_scheme = dx.cf["grid_mapping"].indexing_scheme

        valid_parameters = ("nested", "ring", "nuniq", "zuniq")
        if indexing_scheme not in valid_parameters:
            raise ValueError(
                f"Indexing scheme '{indexing_scheme}' is not in the list of valid parameters: {valid_parameters}\n"
                "Further details: https://cfconventions.org/cf-conventions/cf-conventions.html#healpix"
            )

        return indexing_scheme == "nested"
    except AttributeError:
        # Check legacy HEALPix grid parameters
        indexing_scheme = dx.cf["grid_mapping"].healpix_order

        return indexing_scheme in ["nest", "nested"]


def get_nside(dx):
    try:
        grid_mapping = dx.cf["grid_mapping"]
    except (KeyError, AttributeError):
        # Catch no CF (KeyError) and no grid mapping (AttributeError)
        if dx.squeeze().ndim > 1:
            raise ValueError(
                "Cannot infer the HEALPix resolution from a multidimensional dataset.\n"
                "Consider adding a coordinate reference system to the dataset or passing a one-dimensional array instead.\n"
                "See also: easygems.healpix.attach_coords\n"
                "Reference: https://easy.gems.dkrz.de/Processing/datasets/remapping.html#storing-the-coordinate-reference-system"
            )
        return healpix.npix2nside(dx.size)
    else:
        try:
            return healpix.order2nside(grid_mapping.refinement_level)
        except AttributeError:
            return grid_mapping.healpix_nside


def get_npix(dx):
    return healpix.nside2npix(get_nside(dx))


def get_index_name(dx):
    """Return the name of the (most likely) HEALPix index."""
    for name, coord in dx.coords.items():
        if coord.attrs.get("standard_name") == "healpix_index":
            return name

    for possible_index in ("cell", "value", "values"):
        if possible_index in dx.dims:
            return possible_index

    raise KeyError("Could not find HEALPix index.")


def get_index(dx, dtype=np.int64):
    """Return the (most likely) HEALPix index."""
    healpix_index = get_index_name(dx)
    return xr.DataArray(
        name=healpix_index,
        dims=(healpix_index,),
        data=dx[healpix_index].values.astype(dtype),
        attrs={"standard_name": "healpix_index"},
    )


__all__ = [
    "get_nest",
    "is_nested",
    "get_nside",
    "get_npix",
    "get_index_name",
    "get_index",
]
