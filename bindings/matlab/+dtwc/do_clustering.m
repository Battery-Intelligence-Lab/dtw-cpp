function res = do_clustering(varargin)
%DO_CLUSTERING Run clustering configured by dtwc.settings; returns a dtwc.Result.
%
%   res = dtwc.do_clustering(ds, s)
%   res = dtwc.do_clustering(s)                   % loads s.Input
%   res = dtwc.do_clustering(ds, s, 'Device', 'gpu')
%
%   Writes result files automatically to s.Output ('./results' by default; pass
%   'Output', '' for pure in-memory execution) and honours s.Checkpoint.
%
%   See also dtwc.settings, dtwc.load_data, dtwc.write_data

%> @file do_clustering.m
%> @brief CasADi-style clustering executor.
%> @author Volkan Kumtepeli

    if nargin == 0
        error('dtwc:invalidArgument', 'do_clustering requires data and/or a settings struct.');
    end

    if isstruct(varargin{1}) && isfield(varargin{1}, 'IsDtwcSettings')
        s = dtwc.settings(varargin{1}, varargin{2:end});
        data = dtwc.load_data(s);
    else
        rawData = varargin{1};
        if nargin >= 2 && isstruct(varargin{2}) && isfield(varargin{2}, 'IsDtwcSettings')
            s = dtwc.settings(varargin{2}, varargin{3:end});
        else
            s = dtwc.settings(varargin{2:end});
        end
        data = dtwc.load_data(rawData, s);
    end

    if s.K < 1
        error('dtwc:invalidArgument', 'do_clustering: K must be at least 1.');
    end

    if ischar(data.Source) && data.Ndim == 1 && isempty(dtwc_mex('parquet_files', data.Source))
        out = dtwc_mex('run_workflow', data.Source, {}, double(data.Ndim), ...
                       data.SkipCols, data.SkipRows, data.Delimiter, data.Dtype, ...
                       data.Name, s.Output, s.Checkpoint, s.CheckpointInterval, ...
                       s.K, s.ClusterArgs{:});
    else
        [series, names, effNdim] = data.as_series();
        out = dtwc_mex('run_workflow', cast_series(series, data.Dtype), names, double(effNdim), ...
                       0, 0, '', data.Dtype, data.Name, s.Output, s.Checkpoint, ...
                       s.CheckpointInterval, s.K, s.ClusterArgs{:});
    end
    res = dtwc.Result(out);
end

