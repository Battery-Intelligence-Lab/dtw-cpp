function dx = derivative_transform(x)
%DERIVATIVE_TRANSFORM Apply the Keogh-Pazzani derivative transform.
%
%   dx = dtwc.derivative_transform(x)
%
%   The derivative transform produces a series of the same length that
%   captures the local slope at each point. Used as preprocessing for DDTW.
%
%   Formula (interior points):
%     dx[i] = ((x[i] - x[i-1]) + (x[i+1] - x[i-1]) / 2) / 2
%   Boundary:
%     dx[0]   = x[1] - x[0]
%     dx[n-1] = x[n-1] - x[n-2]
%
%   Parameters
%   ----------
%   x : numeric vector
%       Input time series.
%
%   Returns
%   -------
%   dx : double row vector
%       Derivative-transformed series (same length as input).
%
%   Reference: Keogh & Pazzani (2001), "Derivative Dynamic Time Warping"
%
%   See also dtwc.ddtw_distance

    validateattributes(x, {'numeric'}, {'vector', 'nonempty'}, 'derivative_transform', 'x');
    dx = dtwc_mex('derivative_transform', double(x(:)'));
end
