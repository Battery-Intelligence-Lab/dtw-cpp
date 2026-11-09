function series = cast_series(series, dtype)
%CAST_SERIES Series in memory as the MEX takes them: a numeric matrix, or each
%   numeric element of a cell, cast to double, or to single for DTYPE 'float32'
%   (single data alone stays double); anything else is left as it is, for C++ to
%   refuse naming it. The one conversion behind dtwc.cluster, dtwc.load's
%   'float32' and Problem.set_data.

%> @file cast_series.m
%> @brief Series in memory cast to the class their dtype stores them in.
%> @author Volkan Kumtepeli
    type = 'double';
    if strcmp(dtype, 'float32')
        type = 'single';
    end
    if iscell(series)
        numeric = cellfun(@isnumeric, series);
        series(numeric) = cellfun(@(s) cast(s, type), series(numeric), 'UniformOutput', false);
    elseif isnumeric(series)
        series = cast(series, type);
    end
end
