"""
@file distance.py
@brief Pairwise and cross-collection DTW distances computed in C++/SIMD/CUDA.
@author Volkan Kumtepeli
"""

from __future__ import annotations

import os
import numpy as np

from dtwcpp._api import _float64
from dtwcpp._dtwcpp_core import _cdist, dtw as _dtw


def dtw(x, y, **settings):
    return _dtw(_float64(x), _float64(y), **settings)


# The keywords, their defaults and the refusals are the C++ binding's.
dtw.__doc__ = _dtw.__doc__


def _as_cdist_data(source, *, ndim: int, dtype: str):
    from dtwcpp import Data, InvalidInput, _dtwcpp_core
    from dtwcpp._api import Dataset, _to_data, load_data

    if isinstance(source, Data):
        if ndim != 1 and source.ndim != ndim:
            raise InvalidInput(
                f"cdist: Data already carries ndim={source.ndim}, which disagrees with ndim={ndim}."
            )
        return source
    if isinstance(source, Dataset):
        given = _dtwcpp_core._dtype(dtype)  # a Dataset keeps its own: refuse another, never drop it
        if given not in ("float64", source.dtype):
            raise InvalidInput(
                f"cdist: the Dataset was made with dtype={source.dtype!r} and the settings give dtype={given!r}; "
                "a Dataset keeps the options it was made with, so pass dtype to load_data, or leave it out "
                "of the settings.")
        data = source.as_data()
        if ndim != 1 and data.ndim != ndim:
            raise InvalidInput(
                f"cdist: Dataset carries ndim={data.ndim}, which disagrees with ndim={ndim}."
            )
        return data
    if isinstance(source, (str, os.PathLike)):
        return load_data(source, ndim=ndim, dtype=dtype).as_data()
    return _to_data(source, ndim=ndim, dtype=dtype)


def cdist(
    X,
    Y=None,
    *,
    variant="standard",
    band=-1,
    metric="l1",
    missing_strategy="error",
    wdtw_g=0.05,
    adtw_penalty=1.0,
    sdtw_gamma=1.0,
    msm_c=1.0,
    twe_nu=0.001,
    twe_lambda=1.0,
    mv_mode="dependent",
    ndim=1,
    dtype="float64",
    matrix_dtype="float64",
    device=None,
    gpu_precision="auto",
    gpu_share=-1.0,
):
    """Compute cross-collection ``(n_X, n_Y)`` DTW distance matrix in C++/SIMD/CUDA.

    Parameters
    ----------
    X : Dataset, Data, path, 2-D array, list of 1-D series, DataFrame, or Arrow array
        First collection of ``n_X`` time series (or 3-D ``(n_X, L, ndim)`` when ``ndim > 1``).
    Y : same forms as ``X`` or None, default=None
        Second collection of ``n_Y`` time series. When ``None``, computes ``cdist(X, X)``.
    variant, band, metric, missing_strategy, wdtw_g, adtw_penalty, sdtw_gamma, msm_c, twe_nu, twe_lambda, mv_mode
        DTW family settings (same names and defaults as :func:`dtw` and :func:`dtwcpp.cluster`).
    ndim : int, default=1
        Number of channels per timestep for multivariate series.
    dtype : str, default='float64'
        Storage precision for converted inputs (``'float64'`` or ``'float32'``).
    matrix_dtype : str, default='float64'
        Output NumPy array dtype (``'float64'`` or ``'float32'``).
    device : str or None, default=None
        ``'cpu'``, ``'gpu'``, ``'gpu:N'``, ``'cuda'``, ``'cpu+gpu'``, or ``None``
        (uses :func:`dtwcpp.device`).
    gpu_precision : str, default='auto'
        GPU kernel precision (``'auto'``, ``'fp32'``, or ``'fp64'``).

    Returns
    -------
    numpy.ndarray of shape ``(n_X, n_Y)``
    """
    import dtwcpp

    if isinstance(band, (bool, np.bool_)) or not isinstance(band, (int, np.integer)):
        raise TypeError(f"band must be an integer, got {type(band).__name__}")
    eff_device = dtwcpp.device() if device is None else device
    backend, _ = dtwcpp._resolve_device(eff_device)
    if backend == "hpc":
        raise dtwcpp.DeviceError(
            "device='hpc' offloads clustering jobs to a SLURM cluster and is "
            "not a local compute backend for cdist."
        )

    X_data = _as_cdist_data(X, ndim=ndim, dtype=dtype)
    Y_data = X_data if Y is None else _as_cdist_data(Y, ndim=ndim, dtype=dtype)
    return _cdist(
        X_data,
        Y_data,
        variant=variant,
        band=int(band),
        metric=metric,
        missing_strategy=missing_strategy,
        wdtw_g=float(wdtw_g),
        adtw_penalty=float(adtw_penalty),
        sdtw_gamma=float(sdtw_gamma),
        msm_c=float(msm_c),
        twe_nu=float(twe_nu),
        twe_lambda=float(twe_lambda),
        mv_mode=mv_mode,
        device=eff_device,
        gpu_precision=gpu_precision,
        gpu_share=float(gpu_share),
        matrix_dtype=matrix_dtype,
    )


__all__ = ["dtw", "cdist"]
