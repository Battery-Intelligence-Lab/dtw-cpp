[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'f19_matlab_writeback_evidence.ps1')

$cases = [ordered]@{
    all = @('fast_pam', 'fast_clara', 'clarans', 'cut_dendrogram')
    fast_pam = @('fast_pam')
    fast_clara = @('fast_clara')
    clarans = @('clarans')
    cut_dendrogram = @('cut_dendrogram')
}
$casePasses = 0
$singletonArrays = 0
foreach ($entry in $cases.GetEnumerator()) {
    $actual = Get-F19MatlabRouteNames $entry.Key
    Assert-F19Evidence (
        $actual -is [string[]]
    ) "route '$($entry.Key)' did not return a string array."
    Assert-F19Evidence (
        $actual.Count -eq $entry.Value.Count
    ) "route '$($entry.Key)' returned the wrong cardinality."
    Assert-F19Evidence (
        [string]::Join(',', $actual) -eq
            [string]::Join(',', $entry.Value)
    ) "route '$($entry.Key)' returned the wrong route names."
    $casePasses += 1
    if ($entry.Key -ne 'all') {
        Assert-F19Evidence (
            $actual.Count -eq 1
        ) "route '$($entry.Key)' did not remain a singleton array."
        $singletonArrays += 1
    }
}

$legacyMutantRejections = 0
foreach ($route in @(
    'fast_pam',
    'fast_clara',
    'clarans',
    'cut_dendrogram'
)) {
    $legacyRouteNames = if ($route -eq 'all') {
        @('fast_pam', 'fast_clara', 'clarans', 'cut_dendrogram')
    }
    else {
        @($route)
    }
    if (-not ($legacyRouteNames -is [System.Array])) {
        $legacyMutantRejections += 1
    }
}
Assert-F19Evidence (
    $legacyMutantRejections -eq 4
) (
    "branch-local array mutant rejections=$legacyMutantRejections, " +
    'expected 4.'
)

Write-Output (
    "F19_MATLAB_ROUTE_SELECTOR cases=$casePasses/5 " +
    "singleton_arrays=$singletonArrays/4 " +
    "legacy_mutant_rejections=$legacyMutantRejections/4 verdict=PASS"
)
