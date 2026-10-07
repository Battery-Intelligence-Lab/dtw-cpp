%> @file Dataset.m
%> @brief Lazy dataset handle returned by dtwc.load.
%> @author Volkan Kumtepeli
classdef Dataset < handle
%DATASET A lazy handle to time series: a path, or series already in memory.
%
%   Made by dtwc.load. A path is read once, when its series are first needed
%   (dtwc.cluster, as_series), as dtwc_cl and Python read it: CSV/TSV text and
%   folders of it by the C++ reader (dtwc::read_data, in the MEX); Parquet (a
%   .parquet/.pq file, or a folder of them) by MATLAB's parquetread, by the C++
%   reader's rule (read_parquet). MATLAB has no Arrow IPC reader, so an .arrow,
%   .ipc or .feather file is refused with dtwc:invalidArgument: read it
%   elsewhere and pass its series in memory.
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
                vectors = cellfun(@isvector, series);   % any other element is C++'s to refuse
                lengths = cellfun(@numel, series(vectors));
            else
                series = series(obj.SkipRows + 1:end, :);
                lengths = size(series, 2);
            end
            if any(obj.SkipCols > lengths)
                error('dtwc:invalidArgument', 'load: SkipCols exceeds an in-memory series length.');
            end
            if iscell(series)
                series(vectors) = cellfun(@(s) s(obj.SkipCols + 1:end), series(vectors), ...
                                          'UniformOutput', false);
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
            if ~isempty(delimiter)
                error('dtwc:invalidArgument', ['load: Delimiter splits CSV/TSV text into fields ' ...
                    'and cannot be honoured for a Parquet or Arrow IPC input; drop it.']);
            end
            if arrow
                error('dtwc:invalidArgument', ['load: ''%s'' is Arrow IPC, which MATLAB has no ' ...
                    'reader for. Write the series to Parquet or CSV, or read them elsewhere and ' ...
                    'pass them in memory: a matrix, one series per row, or a cell of vectors.'], path);
            end
            series = {};
            names = {};
            for i = 1:numel(files)
                [s, n] = dtwc.Dataset.read_parquet(files{i}, numel(series), skipCols, skipRows);
                series = [series, s]; %#ok<AGROW>
                names = [names, n]; %#ok<AGROW>
            end
        end

        function [series, names] = read_parquet(file, first, skipCols, skipRows)
        %READ_PARQUET One Parquet file's series, by the C++ reader's rule (dtwc/io/read_data.hpp).
        %   SKIPCOLS drops its leading columns, which then name nothing, and SKIPROWS its leading
        %   rows, which are not read. The first Float32/Float64 column or list column of them
        %   decides: a list column is one series per row; a scalar column is one series, named by
        %   the file, when it is the file's only Float32/Float64 column, and otherwise each row is
        %   a series of its values in the file's columns, string columns aside (any other column
        %   is refused, named). The first string column names the rows, a missing name as
        %   series_<FIRST + row - 1>, as are the rows of a file without one. Only the columns
        %   used are read.
            try
                info = parquetinfo(file);
            catch cause
                error('dtwc:ioError', 'load: failed to read ''%s'': %s', file, cause.message);
            end
            vars = reshape(cellstr(info.VariableNames(min(skipCols, end) + 1:end)), 1, []);
            types = reshape(string(info.VariableTypes(min(skipCols, end) + 1:end)), 1, []);
            floats = ismember(types, ["double", "single"]);
            label = vars(find(types == "string", 1));
            column = '';
            isList = false;
            for i = find(floats | types == "cell")   % a row: vars and types are rows
                if ~floats(i)   % a list column counts when its cells hold floats
                    cells = dtwc.Dataset.read_columns(file, vars(i), 0);
                    if ~all(cellfun(@(v) isfloat(v) || isa(v, 'missing'), cells{1}))
                        continue
                    end
                    isList = true;
                end
                column = vars{i};
                break
            end
            if isempty(column)
                after = '';
                if skipCols > 0
                    after = sprintf(' after the %d columns SkipCols drops', skipCols);
                end
                error('dtwc:ioError', ['load: failed to read ''%s'': no Float32 or Float64 column, ' ...
                      'nor a list column of them%s.'], file, after);
            end
            if ~isList && nnz(floats) == 1          % the file's one float column: one series
                [~, stem] = fileparts(file);
                read = dtwc.Dataset.read_columns(file, {column}, skipRows);
                series = {double(read{1}(:).')};
                names = {stem};
                return
            end
            if isList                               % a series per row of the list column
                read = dtwc.Dataset.read_columns(file, [{column}, label], skipRows);
                nulls = nnz(cellfun(@(v) isa(v, 'missing'), read{1}));
                if nulls
                    error('dtwc:invalidArgument', ['load: ''%s'': list column cell contains %d ' ...
                          'null(s) (drop or fill nulls before clustering).'], file, nulls);
                end
                series = cellfun(@(v) double(v(:).'), read{1}(:).', 'UniformOutput', false);
            else                                    % a series per row of the float columns
                samples = vars(types ~= "string");
                sampleTypes = types(types ~= "string");
                other = find(~ismember(sampleTypes, ["double", "single"]), 1);
                if ~isempty(other)
                    kind = sampleTypes(other);
                    if ismissing(kind)
                        kind = "of a type parquetread cannot read";
                    end
                    error('dtwc:ioError', ['load: failed to read ''%s'': Parquet column ''%s'' is %s, ' ...
                          'neither Float32/Float64 nor Utf8/LargeUtf8, so it cannot be a sample of the ' ...
                          'series each row holds; drop the leading columns with SkipCols, or read ' ...
                          'the column you want with parquetread(file, ''SelectedVariableNames'', name) ' ...
                          'and pass its values to dtwc.load or dtwc.cluster.'], file, samples{other}, kind);
                end
                read = dtwc.Dataset.read_columns(file, [samples, label], skipRows);
                rows = zeros(numel(read{1}), numel(samples));
                for j = 1:numel(samples)
                    rows(:, j) = double(read{j});
                end
                series = num2cell(rows, 2).';
            end
            names = arrayfun(@(i) sprintf('series_%d', i), first + (0:numel(series) - 1), ...
                             'UniformOutput', false);
            if ~isempty(label)
                given = read{end}(:).';
                named = ~ismissing(given);
                names(named) = cellstr(given(named));
            end
        end

        function values = read_columns(file, vars, skipRows)
        %READ_COLUMNS The contents of FILE's columns VARS, in that order, less their first SKIPROWS
        %   rows. Taken by position: parquetread renames a column that is no MATLAB identifier. A
        %   read failure is dtwc:ioError naming the file.
            try
                columns = parquetread(file, 'SelectedVariableNames', vars);
            catch cause
                error('dtwc:ioError', 'load: failed to read ''%s'': %s', file, cause.message);
            end
            values = cell(1, numel(vars));
            for k = 1:numel(vars)
                values{k} = columns{min(skipRows, end) + 1:end, k};
            end
        end
    end
end
