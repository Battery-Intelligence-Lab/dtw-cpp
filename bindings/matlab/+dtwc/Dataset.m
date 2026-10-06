%> @file Dataset.m
%> @brief Lazy dataset handle returned by dtwc.load (api-contract-2.0.md §1.2).
%> @author Volkan Kumtepeli
classdef Dataset < handle
%DATASET A lazy handle to time series: a path, or series already in memory (contract §1.2).
%
%   Made by dtwc.load. A path is read once, when its series are first needed
%   (dtwc.cluster, as_series), as dtwc_cl and Python read it: CSV/TSV text and
%   folders of it by the C++ reader (dtwc::read_data, in the MEX); Parquet (a
%   .parquet/.pq file, or a folder of them) by MATLAB's parquetread, taking
%   what the C++ reader takes: the first Float32/Float64 column (one series,
%   named by its file) or list column of them (one series per row, named
%   series_<i>). MATLAB has no Arrow IPC reader, so an .arrow, .ipc or .feather
%   file is refused with dtwc:invalidArgument: read it elsewhere and pass its
%   series in memory.
%
%   See also dtwc.load, dtwc.cluster

    properties (SetAccess = private)
        Source      % a path, an N x L numeric matrix or a cell of numeric vectors
        SkipCols    % leading fields (a path) or values (in memory) of each series dropped
        SkipRows    % leading lines (a path) or series (in memory) dropped
        Delimiter   % the field delimiter of text; '' infers it from the extension
        Name        % the name of the run and of its result files
    end

    properties (Access = private)
        Series = {}
        Names = {}
        IsRead = false
    end

    methods
        function obj = Dataset(source, skipCols, skipRows, delimiter, name)
        %DATASET Made by dtwc.load, which documents the arguments.
            counts = {'SkipCols', skipCols; 'SkipRows', skipRows};
            for i = 1:2
                v = counts{i, 2};
                if ~(isnumeric(v) && isscalar(v) && isreal(v) && isfinite(v) && v >= 0 && v == fix(v))
                    error('dtwc:invalidArgument', 'load: %s must be a non-negative integer.', counts{i, 1});
                end
            end
            if isstring(source)
                source = char(source);
            end
            if isempty(name)
                path = '';
                if ischar(source)
                    path = source;
                end
                name = dtwc_mex('default_name', path);
            end
            obj.Source = source;
            obj.SkipCols = double(skipCols);
            obj.SkipRows = double(skipRows);
            obj.Delimiter = delimiter;
            obj.Name = name;
        end

        function [series, names] = as_series(obj)
        %AS_SERIES The series, read once. A path gives a cell of double rows and,
        %   in NAMES, the reader's name of each; series in memory are the source
        %   less SkipRows and SkipCols, and NAMES is {} (C++ names them 0, 1, ...).
            if ~obj.IsRead
                if ischar(obj.Source)
                    [obj.Series, obj.Names] = dtwc.Dataset.read_path( ...
                        obj.Source, obj.SkipCols, obj.SkipRows, obj.Delimiter);
                else
                    obj.Series = obj.trimmed();
                end
                obj.IsRead = true;
            end
            series = obj.Series;
            names = obj.Names;
        end
    end

    methods (Access = private)
        function series = trimmed(obj)
        %TRIMMED The in-memory series less the leading SkipRows series and SkipCols values.
            series = obj.Source;
            if (obj.SkipRows == 0 && obj.SkipCols == 0) || ~(iscell(series) || ismatrix(series))
                return   % nothing to drop, or not series: Problem.set_data says why
            end
            if iscell(series)
                series = series(obj.SkipRows + 1:end);
                lengths = cellfun(@numel, series);
            else
                series = series(obj.SkipRows + 1:end, :);
                lengths = size(series, 2);
            end
            if any(obj.SkipCols > lengths)
                error('dtwc:invalidArgument', 'load: SkipCols exceeds an in-memory series length.');
            end
            if iscell(series)
                series = cellfun(@(s) s(obj.SkipCols + 1:end), series, 'UniformOutput', false);
            else
                series = series(:, obj.SkipCols + 1:end);
            end
        end
    end

    methods (Static, Access = private)
        function [series, names] = read_path(path, skipCols, skipRows, delimiter)
        %READ_PATH Every series a path names, read as dtwc::read_data reads it.
            files = dtwc_mex('parquet_files', path);
            [~, ~, ext] = fileparts(path);
            arrow = any(strcmpi(ext, {'.arrow', '.ipc', '.feather'})) && ~isfolder(path);
            if isempty(files) && ~arrow
                [series, names] = dtwc_mex('read_data', path, skipCols, skipRows, delimiter);
                return
            end
            if skipCols || skipRows || ~isempty(delimiter)
                error('dtwc:invalidArgument', ['load: SkipCols, SkipRows and Delimiter parse ' ...
                    'CSV/TSV text and cannot be honoured for a Parquet or Arrow IPC input; drop them.']);
            end
            if arrow
                error('dtwc:invalidArgument', ['load: ''%s'' is Arrow IPC, which MATLAB has no ' ...
                    'reader for. Write the series to Parquet or CSV, or read them elsewhere and ' ...
                    'pass them in memory: a matrix, one series per row, or a cell of vectors.'], path);
            end
            series = {};
            names = {};
            for i = 1:numel(files)
                [s, n] = dtwc.Dataset.read_parquet(files{i}, numel(series));
                series = [series, s]; %#ok<AGROW>
                names = [names, n]; %#ok<AGROW>
            end
        end

        function [series, names] = read_parquet(file, first)
        %READ_PARQUET One Parquet file's series, as the C++ reader takes them: the first
        %   Float32/Float64 column is one series, named by the file; the first list column
        %   of them is one series per row, named series_<FIRST + row - 1>.
            columns = parquetread(file);
            for name = columns.Properties.VariableNames
                values = columns.(name{1});
                if isfloat(values)
                    [~, stem] = fileparts(file);
                    series = {double(values(:).')};
                    names = {stem};
                    return
                end
                if iscell(values) && all(cellfun(@(v) isfloat(v) || isa(v, 'missing'), values))
                    nulls = nnz(cellfun(@(v) isa(v, 'missing'), values));
                    if nulls
                        error('dtwc:invalidArgument', ['load: ''%s'': list column cell contains %d ' ...
                              'null(s) (drop or fill nulls before clustering).'], file, nulls);
                    end
                    series = cellfun(@(v) double(v(:).'), values(:).', 'UniformOutput', false);
                    names = arrayfun(@(i) sprintf('series_%d', i), first + (0:numel(values) - 1), ...
                                     'UniformOutput', false);
                    return
                end
            end
            error('dtwc:ioError', ['load: ''%s'' has no Float32 or Float64 column, nor a list ' ...
                  'column of them.'], file);
        end
    end
end
