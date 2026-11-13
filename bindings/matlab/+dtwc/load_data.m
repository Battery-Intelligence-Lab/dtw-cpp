function ds = load_data(varargin)
%LOAD_DATA Create a lazy dtwc.Dataset configured by dtwc.settings.
%
%   ds = dtwc.load_data('cycles.csv', s)
%   ds = dtwc.load_data(s)                 % reads s.Input
%   ds = dtwc.load_data(X, s)
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
        ds = source;
        return
    end

    ds = dtwc.Dataset(source, s.SkipCols, s.SkipRows, s.Delimiter, s.Name, s.Dtype, s.Ndim);
end

