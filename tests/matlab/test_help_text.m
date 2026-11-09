function tests = test_help_text
%TEST_HELP_TEXT MATLAB's help for every dtwc function and class shows its usage, not a Doxygen header.
%
%   MATLAB's help prints a file's first contiguous comment block. A `%>` Doxygen
%   header placed above `function` or `classdef` is that block, so `help
%   dtwc.cluster` showed "@file cluster.m" instead of the usage. The header now
%   follows the help block, parted from it by a blank line.
%
%   Run with: results = runtests('test_help_text');
%
%   @author Volkan Kumtepeli
tests = functiontests(localfunctions);
end

function test_cluster_help_starts_with_its_usage(testCase)
h = strtrim(help('dtwc.cluster'));
verifyTrue(testCase, startsWith(h, 'CLUSTER'), sprintf('help dtwc.cluster begins: %s', extractBefore([h ' '], ' ')));
verifyTrue(testCase, contains(h, 'dtwc.cluster(data, k)'));
end

function test_a_class_help_starts_with_its_usage(testCase)
h = strtrim(help('dtwc.Result'));
verifyTrue(testCase, startsWith(h, 'RESULT'), sprintf('help dtwc.Result begins: %s', extractBefore([h ' '], ' ')));
end

function test_no_help_shows_a_doxygen_header(testCase)
root = fileparts(which('dtwc.cluster'));          % .../bindings/matlab/+dtwc
files = dir(fullfile(root, '**', '*.m'));
checked = 0;
for i = 1:numel(files)
    if contains(files(i).folder, [filesep 'private']), continue; end  % private functions have no public help
    rel = erase(fullfile(files(i).folder, files(i).name), [fileparts(root) filesep]);
    name = regexprep(erase(rel, '.m'), {'\+', '[\\/]'}, {'', '.'});     % +dtwc/+test/gpu.m -> dtwc.test.gpu
    h = help(name);
    verifyFalse(testCase, contains(h, '@file'), sprintf('help %s shows the Doxygen header', name));
    verifyFalse(testCase, startsWith(strtrim(h), '%>'), sprintf('help %s begins with %%>', name));
    checked = checked + 1;
end
verifyGreaterThan(testCase, checked, 20);  % every public function and class, not a handful
end
