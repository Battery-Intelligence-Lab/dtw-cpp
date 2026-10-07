%> @file load.m
%> @brief Lazy dataset loader.
%> @author Volkan Kumtepeli
function ds = load(source, varargin)
%LOAD A lazy dtwc.Dataset of a path or of series already in memory.
%
%   ds = dtwc.load('cycles.csv', 'SkipCols', 1, 'SkipRows', 1)
%   ds = dtwc.load('cycles/')            % a folder: one series per file
%   ds = dtwc.load('cycles.parquet')     % MATLAB's parquetread
%   ds = dtwc.load(X)                    % N x L numeric matrix, one series per row
%   ds = dtwc.load({x1, x2, x3})         % cell of numeric vectors, any lengths
%
%   SkipCols : leading fields of each line (a path) or values of each series
%              (in memory) dropped. Default 0.
%   SkipRows : leading lines of a file, or leading series in memory (one
%              memory row is one file line). Default 0.
%   Delimiter: the field delimiter of text; '' (default) infers it from the
%              extension (tab for .tsv and .txt, else comma).
%   Name     : the name of the run and of its result files; '' (default) is
%              dtwc_cl's: the file's name without its extension, the folder's
%              name, or 'dataset' for series in memory.
%
%   Nothing is read here: dtwc.cluster (or ds.as_series()) reads a path once.
%   A Dataset passes through unchanged; it keeps the options it was made with.
%
%   See also dtwc.Dataset, dtwc.cluster

    if isa(source, 'dtwc.Dataset')
        if ~isempty(varargin)
            error('dtwc:invalidArgument', ['load: a dtwc.Dataset keeps the options it was ' ...
                  'made with; load its Source with the options instead.']);
        end
        ds = source;
        return
    end
    p = inputParser;
    addParameter(p, 'SkipCols', 0);
    addParameter(p, 'SkipRows', 0);
    addParameter(p, 'Delimiter', '');
    addParameter(p, 'Name', '');
    parse(p, varargin{:});
    ds = dtwc.Dataset(source, p.Results.SkipCols, p.Results.SkipRows, ...
                      char(p.Results.Delimiter), char(p.Results.Name));
end
