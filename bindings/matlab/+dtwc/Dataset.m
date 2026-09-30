%> @file Dataset.m
%> @brief Lazy dataset handle returned by dtwc.load (api-contract-2.0.md §1.2).
%> @author Volkan Kumtepeli
classdef Dataset < handle
%DATASET Lazy dataset handle (contract §1.2).
%
%   A Dataset wraps a data source (an in-memory N x L numeric matrix, a cell
%   array of numeric vectors for ragged series, or a file path) plus load
%   options. It performs NO file I/O at construction time: a path is only
%   read by dtwc.cluster (C++ dtwc::cluster). This preserves the large-N
%   lazy-load contract.
%
%   Create via dtwc.load(...), not directly.
%
%   See also dtwc.load, dtwc.cluster

    properties (SetAccess = private)
        Source          % numeric N x L matrix, cell of series, OR char/string path
        SkipCols double = 0
        SkipRows double = 0
        Delimiter char = ''
        Name char = 'dataset'
    end

    methods
        function obj = Dataset(source, skip_cols, skip_rows, delimiter, name)
        %DATASET Construct a lazy dataset handle.
            if nargin >= 1, obj.Source = source; end
            if nargin >= 2, obj.SkipCols = double(skip_cols); end
            if nargin >= 3, obj.SkipRows = double(skip_rows); end
            if nargin >= 4, obj.Delimiter = char(delimiter); end
            if nargin >= 5, obj.Name = char(name); end
        end
    end

    methods (Static, Hidden)
        function out = normalise_cell_series(source, caller)
        %NORMALISE_CELL_SERIES 1xN cell of real double ROW vectors, or raise.
        %   The MEX gateway's cell_to_series requires real, non-empty, non-sparse
        %   doubles; validating here keeps the taxonomy dtwc:invalidArgument.
            out = reshape(source, 1, []);
            for i = 1:numel(out)
                v = out{i};
                if ~isnumeric(v) || ~isreal(v) || issparse(v) || isempty(v) || ~isvector(v)
                    error('dtwc:invalidArgument', ...
                          '%s: data{%d} must be a non-empty real numeric vector.', ...
                          caller, i);
                end
                out{i} = double(v(:)).';
            end
        end
    end
end
