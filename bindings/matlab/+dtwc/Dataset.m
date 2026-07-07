%> @file Dataset.m
%> @brief Lazy dataset handle returned by dtwc.load (api-contract-2.0.md §1.2).
%> @author Volkan Kumtepeli
classdef Dataset < handle
%DATASET Lazy dataset handle (contract §1.2).
%
%   A Dataset wraps a data source (an in-memory N x L numeric matrix or a file
%   path) plus load options. It performs NO file I/O at construction time — a
%   path is only read when materialize() is called (from dtwc.cluster). This
%   preserves the large-N / device='hpc' lazy-load contract.
%
%   Create via dtwc.load(...), not directly.
%
%   See also dtwc.load, dtwc.cluster

    properties (SetAccess = private)
        Source          % numeric N x L matrix OR char/string file path
        SkipCols double = 0
        Delimiter char = ''
        Name char = 'dataset'
    end

    methods
        function obj = Dataset(source, skip_cols, delimiter, name)
        %DATASET Construct a lazy dataset handle.
            if nargin >= 1, obj.Source = source; end
            if nargin >= 2, obj.SkipCols = double(skip_cols); end
            if nargin >= 3, obj.Delimiter = char(delimiter); end
            if nargin >= 4, obj.Name = char(name); end
        end

        function [X, names] = materialize(obj)
        %MATERIALIZE Resolve the source into an N x L double matrix (+ names).
        %   [X, names] = ds.materialize()
        %
        %   For an in-memory matrix source this is a no-op cast. For a path
        %   source the file is read HERE (lazily), honouring SkipCols/Delimiter.
            names = {};
            if isnumeric(obj.Source)
                X = double(obj.Source);
            elseif ischar(obj.Source) || isstring(obj.Source)
                path = char(obj.Source);
                if isempty(obj.Delimiter)
                    X = readmatrix(path);
                else
                    X = readmatrix(path, 'Delimiter', obj.Delimiter);
                end
                if obj.SkipCols > 0
                    X = X(:, (obj.SkipCols + 1):end);
                end
            else
                error('dtwc:invalidArgument', ...
                      'Dataset source must be a numeric matrix or a file path.');
            end
        end
    end
end
