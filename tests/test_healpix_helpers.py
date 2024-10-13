import numpy as np
from easygems.healpix import get_contiguous_chunks


def test_contiguous_chunks():
    assert get_contiguous_chunks([1], chunksize=1) == 1
    assert np.array_equal(
        get_contiguous_chunks([1], chunksize=4), np.array([0, 1, 2, 3])
    )
    assert np.array_equal(
        get_contiguous_chunks([1, 15], chunksize=4),
        np.array([0, 1, 2, 3, 12, 13, 14, 15]),
    )
