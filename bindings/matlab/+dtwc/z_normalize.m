%> @file z_normalize.m
%> @brief Z-normalize a time series (zero mean, unit std dev).
%> @author Volkan Kumtepeli
function xn = z_normalize(x)
%Z_NORMALIZE Z-normalize a time series (zero mean, unit std dev).
%
%   xn = dtwc.z_normalize(x)
%
%   Subtracts the mean and divides by the standard deviation. If the
%   standard deviation is below 1e-10, all values are set to zero.
%
%   Parameters
%   ----------
%   x : numeric vector
%       Input time series.
%
%   Returns
%   -------
%   xn : double row vector
%       Z-normalized series.
%
%   See also dtwc.derivative_transform

    validateattributes(x, {'numeric'}, {'vector', 'nonempty'}, 'z_normalize', 'x');
    xn = dtwc_mex('z_normalize', double(x(:)'));
end
