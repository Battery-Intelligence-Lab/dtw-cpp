function D = cdist(X, varargin)
%CDIST Top-level alias for dtwc.distance.cdist(X, ...).
%
%   See also dtwc.distance.cdist

%> @file cdist.m
%> @brief Top-level alias for dtwc.distance.cdist.
%> @author Volkan Kumtepeli

    D = dtwc.distance.cdist(X, varargin{:});
end

