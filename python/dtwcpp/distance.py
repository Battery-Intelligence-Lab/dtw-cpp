"""
@file distance.py
@brief The DTW-family distance of two series, computed by C++ (dtwc::distance::dtw).
"""

from __future__ import annotations

import numpy as np

from dtwcpp._dtwcpp_core import dtw as _dtw


def dtw(x, y, **settings):
    return _dtw(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64),
                **settings)


# The keywords, their defaults and the refusals are the C++ binding's.
dtw.__doc__ = _dtw.__doc__

__all__ = ["dtw"]
