function ds = load_data(varargin)
%LOAD_DATA Create a lazy dtwc.Dataset configured by dtwc.settings.
%
%   ds = dtwc.load_data('cycles.csv', s)
%   ds = dtwc.load_data(s)                 % reads s.Input
%   ds = dtwc.load_data(X, s)
%
%   A dtwc.Dataset passes through with the reader options it was made with:
%   settings that give one of them (SkipCols, SkipRows, Delimiter, Dtype, Ndim)
%   a value other than its default and the Dataset's are refused.
%
%   See also dtwc.settings, dtwc.do_clustering, dtwc.write_data

%> @file load_data.m
%> @brief CasADi-style dataset loader.
%> @author Volkan Kumtepeli

    if nargin == 0
        error('dtwc:invalidArgument', 'load_data requires a source or a settings struct with Input set.');
    end

    if isstruct(varargin{1}) && isfield(varargin{1}, 'IsDtwcSettings')
        s = dtwc.settings(varargin{1}, varargin{2:end});
        if isempty(s.Input)
            error('dtwc:invalidArgument', 'load_data: pass a data source or set Input in dtwc.settings.');
        end
        source = s.Input;
    else
        source = varargin{1};
        if nargin >= 2 && isstruct(varargin{2}) && isfield(varargin{2}, 'IsDtwcSettings')
            s = dtwc.settings(varargin{2}, varargin{3:end});
        else
            s = dtwc.settings(varargin{2:end});
        end
    end

    if isa(source, 'dtwc.Dataset')
        % It is read with its own reader options: one the settings give, not at its default, must be the same.
        defaults = {'SkipCols', 0; 'SkipRows', 0; 'Delimiter', ''; 'Dtype', 'float64'; 'Ndim', 1};
        for i = 1:size(defaults, 1)
            key = defaults{i, 1};
            if ~same(s.(key), defaults{i, 2}) && ~same(s.(key), source.(key))
                error('dtwc:invalidArgument', ['load_data: the Dataset was made with %s %s and the settings ' ...
                      'give %s %s; a Dataset keeps the options it was made with, so pass %s to load_data, ' ...
                      'or leave it out of the settings.'], key, mat2str(source.(key)), key, mat2str(s.(key)), key);
            end
        end
        ds = source;
        return
    end

    ds = dtwc.Dataset(source, s.SkipCols, s.SkipRows, s.Delimiter, s.Name, s.Dtype, s.Ndim);
end

function tf = same(a, b)
%SAME Whether two option values are one: equal in value whatever their class, or both empty ('' and "").
    tf = isequal(a, b) || (isempty(a) && isempty(b));
end

