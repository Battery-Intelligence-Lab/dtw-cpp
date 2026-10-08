"""
@file _mip.py
@brief Method.MIP, and Method.LRCore's kelley root, with the installed highspy, for an
       extension that links no HiGHS (the wheel).
@details
    C++ builds the compact p-median model as arrays (dtwc::mip::build_p_median_model,
    the model dtwc_cl and MATLAB hand to their linked HiGHS); highspy reads them as
    NumPy views, and C++ decodes the solution into labels and medoids. LR-core runs in
    C++ and solves its Kelley root's master LP through _KelleyMaster.
@author Volkan Kumtepeli
"""


def cluster(prob):
    """``Problem.cluster()`` for Method.MIP on the HiGHS solver: solve prob's model with highspy.

    The MIPSettings that apply are mapped as C++ maps them for linked HiGHS (``mip_gap``,
    ``time_limit_sec``, ``verbose_solver``, the FastPAM ``warm_start`` as the MIP start),
    and HiGHS runs on the threads OpenMP gives dtwcpp. Without highspy, ``SolverError``
    names the ``mip`` extra; a solve that does not reach optimality is ``SolverError``.
    """
    import numpy as np

    from dtwcpp import SolverError, _dtwcpp_core
    try:
        import highspy
    except ImportError as error:  # absent, or present and failing to import: the message says which
        raise SolverError(f"method 'mip' solves with highspy, which cannot be imported ({error}): "
                          "pip install \"dtwcpp[mip]\". Method 'lrcore' is exact without it.") from None

    model = _dtwcpp_core._mip_model(prob)  # checks k, N and the settings; fills the matrix
    settings = prob.mip_settings
    highs = highspy.Highs()
    options = {"output_flag": settings.verbose_solver, "mip_rel_gap": settings.mip_gap,
               "threads": _dtwcpp_core.openmp_max_threads()}
    if settings.time_limit_sec > 0:
        options["time_limit"] = float(settings.time_limit_sec)
    for name, value in options.items():  # a rejected option must not leave HiGHS's default
        if highs.setOptionValue(name, value) == highspy.HighsStatus.kError:
            raise SolverError(f"highspy rejected option '{name}' = {value!r}; continuing would "
                              "silently run a different solver configuration.")
    status = highs.passModel(
        model.num_col, model.num_row, model.a_value.size, int(highspy.MatrixFormat.kRowwise),
        int(highspy.ObjSense.kMinimize), 0.0, model.col_cost, model.col_lower, model.col_upper,
        model.row_lower, model.row_upper, model.a_start, model.a_index, model.a_value,
        model.integrality)
    if status != highspy.HighsStatus.kOk:
        raise SolverError(f"highspy rejected the MIP model (passModel returned {status}).")
    if settings.warm_start:
        start = highspy.HighsSolution()
        start.col_value = model.start
        start.value_valid = True
        highs.setSolution(start)

    # HiGHS keeps one scheduler per calling thread and refuses a run whose thread
    # count differs from it: the run starts from none and leaves none.
    highspy.Highs.resetGlobalScheduler(True)
    try:
        status = highs.run()
    finally:
        highspy.Highs.resetGlobalScheduler(True)
    if status != highspy.HighsStatus.kOk:
        raise SolverError(f"highspy failed to solve the MIP (run returned {status}).")
    model_status = highs.getModelStatus()
    if model_status != highspy.HighsModelStatus.kOptimal:
        raise SolverError("highspy MIP did not solve to optimality. Model status: "
                          + highs.modelStatusToString(model_status))
    return _dtwcpp_core._mip_set_solution(prob, np.asarray(highs.getSolution().col_value))


class _KelleyMaster:
    """The master LP of LR-core's Kelley root (dtwc::mip::KelleyMaster) on the installed
    highspy, as linked HiGHS solves it in C++: maximise over box-bounded columns, one row
    per cut, re-solved after the column bounds move. C++ calls each method."""

    def __init__(self, highspy):
        self._highspy = highspy
        self._highs = highspy.Highs()
        self._highs.setOptionValue("output_flag", False)

    def pass_model(self, cost, lower, upper):
        lp = self._highspy.HighsLp()
        lp.num_col_ = len(cost)
        lp.sense_ = self._highspy.ObjSense.kMaximize
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, lower, upper
        lp.a_matrix_.num_col_ = len(cost)
        lp.a_matrix_.start_ = [0] * (len(cost) + 1)  # no rows yet
        return self._highs.passModel(lp) == self._highspy.HighsStatus.kOk

    def add_row(self, lower, upper, index, value):
        self._highs.addRow(lower, upper, len(index), index, value)

    def change_col_bounds(self, first, lower, upper):
        import numpy as np
        columns = np.arange(first, first + len(lower), dtype=np.int32)
        self._highs.changeColsBounds(len(columns), columns, lower, upper)

    def run(self):
        """The column values of an optimal solve, else None."""
        if (self._highs.run() != self._highspy.HighsStatus.kOk
                or self._highs.getModelStatus() != self._highspy.HighsModelStatus.kOptimal):
            return None
        return self._highs.getSolution().col_value


def lrcore_kelley(prob):
    """``Problem.cluster()`` for Method.LRCore with ``lr_root="kelley"`` on an extension
    that links no HiGHS: LR-core runs in C++, and its Kelley root's master LP solves with
    highspy. Without highspy, ``SolverError`` names the ``mip`` extra before any work."""
    from dtwcpp import SolverError, _dtwcpp_core
    try:
        import highspy
    except ImportError as error:  # absent, or present and failing to import: the message says which
        raise SolverError(f"lr_root 'kelley' solves its master LP with highspy, which cannot be imported "
                          f"({error}): pip install \"dtwcpp[mip]\", or keep lr_root 'subgradient', the "
                          "default, which needs no solver.") from None
    # As in cluster(): the run starts from no HiGHS scheduler and leaves none.
    highspy.Highs.resetGlobalScheduler(True)
    try:
        return _dtwcpp_core._lr_core(prob, _KelleyMaster(highspy))
    finally:
        highspy.Highs.resetGlobalScheduler(True)
