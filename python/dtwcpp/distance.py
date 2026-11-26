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


def _as_cdist_data(source, *, ndim: int | None, dtype):
    from dtwcpp import Data, InvalidInput, _dtwcpp_core
    from dtwcpp._api import Dataset, _to_data

    if ndim is not None:
        if isinstance(ndim, (bool, np.bool_)) or not isinstance(ndim, (int, np.integer)):
            raise TypeError(f"ndim must be an integer, got {type(ndim).__name__}")
        if int(ndim) < 1:
            raise InvalidInput("ndim must be at least 1.")
        ndim = int(ndim)

    if isinstance(source, (Data, Dataset)):  # each keeps its own dtype: refuse another, never drop it
        kind, own = (("Dataset", source.dtype) if isinstance(source, Dataset)
                     else ("Data", "float32" if source.is_f32() else "float64"))
        given = own if dtype is None else _dtwcpp_core._dtype(dtype)
        if given != own:
            raise InvalidInput(
                f"cdist: the {kind} was made with dtype={own!r} and dtype={given!r} was given; "
                f"a {kind} keeps the options it was made with, so pass dtype to load, or leave it out.")
    if isinstance(source, Data):
        if ndim is not None and source.ndim != ndim:
            raise InvalidInput(
                f"cdist: Data already carries ndim={source.ndim}, which disagrees with ndim={ndim}."
            )
        return source
    if isinstance(source, Dataset):
        data = source.as_data()
        if ndim is not None and data.ndim != ndim:
            raise InvalidInput(
                f"cdist: Dataset carries ndim={data.ndim}, which disagrees with ndim={ndim}."
            )
        return data
    dtype = "float64" if dtype is None else dtype
    if isinstance(source, (str, os.PathLike)):
        data = Dataset(source, dtype=dtype, ndim=ndim).as_data()
        if ndim is not None and data.ndim != ndim:
            raise InvalidInput(
                f"cdist: Dataset carries ndim={data.ndim}, which disagrees with ndim={ndim}."
            )
        return data
    return _to_data(source, ndim=1 if ndim is None else ndim, dtype=dtype)


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
    ndim=None,
    dtype=None,
    matrix_dtype="float64",
    device=None,
    gpu_precision="auto",
    gpu_share=-1.0,
    fallback=False,
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
    ndim : int or None, default=None
        Number of channels per timestep for multivariate series (None: a Dataset/Data/Arrow
        IPC file's own ndim, else 1).
    dtype : str or None, default=None
        Storage precision for converted inputs (``'float64'`` or ``'float32'``; None: ``'float64'``).
        A Dataset or Data keeps its own: a dtype other than its own raises :class:`dtwcpp.InvalidInput`.
    matrix_dtype : str, default='float64'
        Output NumPy array dtype (``'float64'`` or ``'float32'``).
    device : str or None, default=None
        ``'cpu'``, ``'gpu'``, ``'gpu:N'``, ``'cuda'``, ``'cpu+gpu'``, or ``None``
        (uses :func:`dtwcpp.device`).
    gpu_precision : str, default='auto'
        GPU kernel precision (``'auto'``, ``'fp32'``, or ``'fp64'``).
    fallback : bool, default=False
        A GPU request that cannot be honoured (no GPU backend or device, a variant,
        missing-data strategy, precision or number of features the kernels lack) runs
        on the CPU, with one line on stderr saying why, instead of raising
        :class:`dtwcpp.DeviceError`. For this call only: :func:`dtwcpp.device` has no
        fallback.

    Returns
    -------
    numpy.ndarray of shape ``(n_X, n_Y)``
    """
    import dtwcpp

    if isinstance(band, (bool, np.bool_)) or not isinstance(band, (int, np.integer)):
        raise TypeError(f"band must be an integer, got {type(band).__name__}")
    if not isinstance(fallback, (bool, np.bool_)):
        raise TypeError(f"fallback must be a bool, got {type(fallback).__name__}")
    eff_device = dtwcpp.device() if device is None else device
    backend, _ = dtwcpp._resolve_device(eff_device, fallback)
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
        fallback=bool(fallback),
    )


__all__ = ["dtw", "cdist"]
