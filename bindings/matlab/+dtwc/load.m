function ds = load(source, varargin)
%LOAD A lazy dtwc.Dataset of a path or of series already in memory.
%
%   ds = dtwc.load('cycles.csv', 'SkipCols', 1, 'SkipRows', 1)
%   ds = dtwc.load('cycles.csv', 'Dtype', 'float32')   % stored as single
%   ds = dtwc.load('cycles/')            % a folder: one series per file
%   ds = dtwc.load('cycles.parquet')     % MATLAB's parquetread
%   ds = dtwc.load(X)                    % N x L numeric matrix, one series per row
%   ds = dtwc.load({x1, x2, x3})         % cell of numeric vectors, any lengths
%
%   SkipCols : leading fields of each line, or columns of a Parquet file (a
%              path), or values of each series (in memory) dropped. Default 0.
%   SkipRows : leading lines of a file, rows of a Parquet file, or leading
%              series in memory (one memory row is one file line). Default 0.
%   Delimiter: the field delimiter of text; '' (default) infers it from the
%              extension (tab for .tsv and .txt, else comma).
%   Name     : the name of the run and of its result files; '' (default) is
%              dtwc_cl's: the file's name without its extension, the folder's
%              name, or 'dataset' for series in memory.
%   Dtype    : the precision the series are stored in: 'float64' (default,
%              double) or 'float32' (single, half the memory; also 'f32', 'fp32',
%              'float'). Each value is read as a double and rounded to single;
%              single data in memory alone does not choose it.
%
%   Nothing is read here. ds.as_series() reads a path once and keeps its series,
%   which dtwc.cluster takes for Parquet; dtwc.cluster hands any other path to C++
%   dtwc::run, which reads it at every call (and may stream an Arrow IPC file
%   under RamLimit); to read it once, cluster its series: dtwc.cluster(ds.as_series(), k).
%   A Dataset passes through with the options it was made with: a SkipCols,
%   SkipRows, Delimiter or Dtype given here that differs from the Dataset's, at
%   its default too, is refused, naming both values; one equal to it runs, and a
%   Name is refused.
%
%   See also dtwc.Dataset, dtwc.cluster

%> @file load.m
%> @brief Lazy dataset loader.
%> @author Volkan Kumtepeli

    p = inputParser;
    addParameter(p, 'SkipCols', 0);
    addParameter(p, 'SkipRows', 0);
    addParameter(p, 'Delimiter', '');
    addParameter(p, 'Name', '');
    addParameter(p, 'Dtype', 'float64');
    parse(p, varargin{:});
    if isa(source, 'dtwc.Dataset')
        if ~ismember('Name', p.UsingDefaults)
            error('dtwc:invalidArgument', 'load: Dataset keeps the options it was made with; unexpected Name.');
        end
        for key = {'SkipCols', 'SkipRows', 'Delimiter', 'Dtype'}
            if ismember(key{1}, p.UsingDefaults)
                continue
            end
            given = p.Results.(key{1});
            if strcmp(key{1}, 'Dtype')
                given = dtwc_mex('dtype', char(given));   % the C++ table's canonical name, as a Dataset holds it
            elseif strcmp(key{1}, 'Delimiter')
                given = char(given);
            end
            if ~same(given, source.(key{1}))
                error('dtwc:invalidArgument', ['load: the Dataset was made with %s %s and %s %s was given; ' ...
                      'a Dataset keeps the options it was made with, so pass %s to dtwc.load, or leave it out.'], ...
                      key{1}, mat2str(source.(key{1})), key{1}, mat2str(given), key{1});
            end
        end
        ds = source;
        return
    end
    ds = dtwc.Dataset(source, p.Results.SkipCols, p.Results.SkipRows, ...
                      char(p.Results.Delimiter), char(p.Results.Name), char(p.Results.Dtype));
end

function tf = same(a, b)
%SAME Whether two option values are one: equal in value whatever their class, or both empty ('' and "").
    tf = isequal(a, b) || (isempty(a) && isempty(b));
end
