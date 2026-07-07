"""
@file test.py
@brief ``dtwcpp.test`` — capability self-introspection (Task 3.3).

Thin, dependency-free wrappers over the C++ ``dtwc::test::*`` probes bound in the
core extension. Both return plain ``dict`` results whose keys are IDENTICAL to the
C++ struct fields and the MATLAB ``dtwc_mex('test_*')`` struct fields, so the same
introspection code reads the same names in C++, Python and MATLAB.

Examples
--------
>>> import dtwcpp
>>> dtwcpp.test.parallelisation()
{'available': True, 'max_threads': 24, 'threads_engaged': 24, 'pass': True, 'reason': ''}
>>> dtwcpp.test.gpu()['available']
False
"""

from dtwcpp._dtwcpp_core import (
    test_parallelisation as _test_parallelisation,
    test_gpu as _test_gpu,
)


def parallelisation():
    """Run a real OpenMP parallel region and report engaged threads.

    Returns
    -------
    dict
        ``{available, max_threads, threads_engaged, pass, reason}``.
        ``threads_engaged`` counts DISTINCT OpenMP thread ids that actually
        executed (proof-of-engagement, not a compile-flag read). On a sequential
        build ``available`` is ``False`` and ``reason`` is a non-empty string.
    """
    return _test_parallelisation()


def gpu():
    """Execute a tiny GPU kernel and validate it against a CPU oracle.

    Returns
    -------
    dict
        ``{available, backend, device_name, validated, pass, reason}``.
        When a GPU backend is compiled in and a device is present, the kernel is
        run and ``validated`` reports whether it matched the CPU oracle within
        tolerance. Otherwise ``available`` is ``False`` and ``reason`` names
        exactly what is missing. Never raises, never silently degrades.
    """
    return _test_gpu()


__all__ = ["parallelisation", "gpu"]
