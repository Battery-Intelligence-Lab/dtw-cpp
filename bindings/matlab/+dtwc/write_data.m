function outdir = write_data(res, directory)
%WRITE_DATA Write a dtwc.Result's files and run report into DIRECTORY.
%
%   dtwc.write_data(res)                % writes into './results'
%   dtwc.write_data(res, './archive')
%
%   See also dtwc.settings, dtwc.load_data, dtwc.do_clustering, dtwc.Result

%> @file write_data.m
%> @brief CasADi-style result writer.
%> @author Volkan Kumtepeli

    if nargin < 2 || isempty(directory)
        directory = './results';
    end
    if ~isa(res, 'dtwc.Result')
        error('dtwc:invalidArgument', 'write_data: first argument must be a dtwc.Result.');
    end
    outdir = char(directory);
    res.save(outdir);
end

