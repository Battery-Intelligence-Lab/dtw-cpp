%> @file load.m
%> @brief Lazy dataset loader (api-contract-2.0.md §1.2).
%> @author Volkan Kumtepeli
function ds = load(source, varargin)
%LOAD Create a lazy dtwc.Dataset handle (contract §1.2).
%
%   ds = dtwc.load(source)
%   ds = dtwc.load(source, 'skip_cols', 0, 'delimiter', '', 'name', '')
%
%   Parameters
%   ----------
%   source : char/string path OR N x L numeric matrix
%       Each matrix row is one time series of length L.
%   skip_cols : leading columns to drop (id columns). Default 0.
%   delimiter : field delimiter for path sources. '' = auto from extension.
%   name : dataset name. '' = derive from the filename stem (or 'dataset').
%
%   Contract: load() performs NO file I/O. A path source is only read when the
%   dataset is materialised inside dtwc.cluster().
%
%   See also dtwc.Dataset, dtwc.cluster

    p = inputParser;
    addRequired(p, 'source');
    addParameter(p, 'skip_cols', 0, @(v) isnumeric(v) && isscalar(v) && v >= 0);
    addParameter(p, 'delimiter', '', @(v) ischar(v) || isstring(v));
    addParameter(p, 'name', '', @(v) ischar(v) || isstring(v));
    parse(p, source, varargin{:});

    nm = char(p.Results.name);
    if isempty(nm)
        if ischar(source) || isstring(source)
            [~, nm, ~] = fileparts(char(source));
        else
            nm = 'dataset';
        end
    end

    ds = dtwc.Dataset(source, double(p.Results.skip_cols), ...
                      char(p.Results.delimiter), nm);
end
