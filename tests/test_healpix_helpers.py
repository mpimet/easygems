import numpy as np
from easygems.healpix import get_full_chunks


def test_full_chunks():
    assert get_full_chunks([1], chunksize=1) == 1
    assert np.array_equal(
        get_full_chunks([1], chunksize=4), np.array([0, 1, 2, 3])
    )
    assert np.array_equal(
        get_full_chunks([1, 15], chunksize=4),
        np.array([0, 1, 2, 3, 12, 13, 14, 15]),
    )
