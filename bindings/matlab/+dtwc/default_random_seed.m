%> @file default_random_seed.m
%> @brief Shared invocation-local seed used by deterministic Tier-1 algorithms.
function seed = default_random_seed()
%DEFAULT_RANDOM_SEED Return the cross-language Tier-1 random seed (42).
%   The value comes from dtwc::settings::DEFAULT_RANDOM_SEED in the C++ core;
%   it is not a second MATLAB-side literal.
    seed = dtwc_mex('default_random_seed');
end
