function tests = test_distance
%TEST_DISTANCE dtwc.distance.dtw, the one distance entry, against values worked out by hand.
%   Every variant x metric x missing-data strategy that C++ computes (Standard
%   with every metric and strategy, DDTW with every metric, the other variants
%   with L1), each variant's parameter away from its default; then one row per
%   refusal of core::validate, of the input scan and of the name-value reader.
%   The values are tests/python/test_distance.py's, derived there.
    tests = functiontests(localfunctions);
end

function test_distance_matches_the_hand_value(testCase)
    gx = [0 NaN 2];
    gy = [0 9 9 0];
    sq = {'Metric', 'squared_euclidean'};
    rows = {
        {},                                       [1 2 3],     [3 4 5 6 7],     13
        sq,                                       [1 2 3],     [3 4 5 6 7],     35
        {'Band', 2},                              [0 1 0 2 0], [0 0 0 0 0 0 2], 5
        {'MissingStrategy', 'zero_cost'},         gx,          gy,              2
        [{'MissingStrategy', 'zero_cost'}, sq],   gx,          gy,              4
        {'MissingStrategy', 'arow'},              gx,          gy,              9
        [{'MissingStrategy', 'arow'}, sq],        gx,          gy,              53
        {'MissingStrategy', 'interpolate'},       gx,          gy,              17
        [{'MissingStrategy', 'interpolate'}, sq], gx,          gy,              103
        {'Variant', 'ddtw'},                      [0 1 2],     [0 3 6],         6
        [{'Variant', 'ddtw'}, sq],                [0 1 2],     [0 3 6],         12
        {'Variant', 'wdtw', 'WdtwG', 0},          [1 2 3],     [3 4 5 6 7],     6.5
        {'Variant', 'adtw', 'AdtwPenalty', 2},    [0 1],       [0 0 1],         2
        {'Variant', 'softdtw', 'SdtwGamma', 0.7}, [0 0],       [0 0],           -0.7 * log(3)
        {'Variant', 'msm', 'MsmC', 0.5},          [0 2],       1,               2.5
        {'Variant', 'twe', 'TweNu', 0.05, 'TweLambda', 0.6}, [0 0], 0,          0.65
    };
    for i = 1:size(rows, 1)
        [settings, x, y, want] = rows{i, :};
        verifyEqual(testCase, dtwc.distance.dtw(x, y, settings{:}), want, ...
            'RelTol', 1e-15, sprintf('row %d', i));
    end
    % name=value syntax passes each name as a string scalar.
    verifyEqual(testCase, dtwc.distance.dtw([0 1 0 2 0], [0 0 0 0 0 0 2], Band=2), 5);
end

function test_distance_refuses_what_no_kernel_computes(testCase)
    rows = {
        {'Variant', 'bogus'}, 0, 0, ...
            'unknown variant ''bogus''. Valid: standard, ddtw, wdtw, adtw, softdtw, msm, twe.'
        {'WdtwG', -1}, 0, 0, 'WDTW g must be finite and non-negative.'
        {'Variant', 'ddtw', 'MissingStrategy', 'zero_cost'}, 0, 0, ...
            'Non-Standard DTW variants require MissingStrategy::Error.'
        {'Variant', 'wdtw', 'Metric', 'squared_euclidean'}, 0, 0, ...
            ['metric SquaredL2 is implemented for Standard DTW and DDTW only, but variant = ' ...
             'wdtw was requested. Use metric L1 for this configuration.']
        {}, [0 NaN 2], [0 9 9 0], 'x[1] is NaN'
        {'MissingStrategy', 'zero_cost'}, [0 NaN 2], [0 9 Inf 0], 'y[2] is +inf'
        {'Bogus', 1}, 0, 0, ...
            ['unknown distance setting ''Bogus''. Valid: Variant, Band, Metric, ' ...
             'MissingStrategy, WdtwG, AdtwPenalty, SdtwGamma, MsmC, TweNu, TweLambda.']
    };
    for i = 1:size(rows, 1)
        [settings, x, y, message] = rows{i, :};
        err = [];
        try
            dtwc.distance.dtw(x, y, settings{:});
        catch caught
            err = caught;
        end
        assertNotEmpty(testCase, err, sprintf('row %d was not refused', i));
        verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument', sprintf('row %d', i));
        verifySubstring(testCase, err.message, message, sprintf('row %d', i));
    end
end
