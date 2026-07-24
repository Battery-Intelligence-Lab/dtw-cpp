[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $MexBin,

    [Parameter(Mandatory = $true)]
    [string] $SourceSnapshot,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9A-Fa-f]{64}$')]
    [string] $ExpectedMexSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9A-Fa-f]{64}$')]
    [string] $ExpectedSourceSha256,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')]
    [string] $Profile,

    [ValidateSet('all', 'fast_pam', 'fast_clara', 'clarans', 'cut_dendrogram')]
    [string] $Route = 'all',

    [ValidateSet('both', 'R2024b', 'R2025b')]
    [string] $Release = 'both',

    [string] $MatlabR2024b = '',
    [string] $MatlabR2025b = '',
    [string] $RuntimeDirectory = 'build\f19-matlab-writeback',

    [Parameter(Mandatory = $true)]
    [string] $SummaryPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'f19_matlab_writeback_evidence.ps1')

$script:MinimumVectorAssertions = 8
$script:ScalarAssertionsPerExecution = 5

function Assert-F19 {
    param(
        [Parameter(Mandatory = $true)]
        [bool] $Condition,

        [Parameter(Mandatory = $true)]
        [string] $Message
    )

    if (-not $Condition) {
        throw "F19 MATLAB writeback gate: $Message"
    }
}

function Get-F19FullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [string] $BasePath
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

function Assert-F19PathInsideRepository {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [string] $RepositoryRoot,

        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedRoot = [System.IO.Path]::GetFullPath($RepositoryRoot)
    $rootPrefix = $resolvedRoot.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    Assert-F19 (
        $resolvedPath.StartsWith(
            $rootPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) "$Description must resolve inside the repository: $resolvedPath"
}

function New-F19Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [string] $RepositoryRoot
    )

    Assert-F19PathInsideRepository $Path $RepositoryRoot 'runtime directory'
    if (-not [System.IO.Directory]::Exists($Path)) {
        [void][System.IO.Directory]::CreateDirectory($Path)
    }
}

function Write-F19Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string] $Content,

        [Parameter(Mandatory = $true)]
        [string] $RepositoryRoot
    )

    Assert-F19PathInsideRepository $Path $RepositoryRoot 'runtime artifact'
    $parent = [System.IO.Path]::GetDirectoryName(
        [System.IO.Path]::GetFullPath($Path)
    )
    New-F19Directory $parent $RepositoryRoot
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function ConvertTo-F19MatlabLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Value
    )

    $portable = $Value.Replace('\', '/').Replace("'", "''")
    return "'$portable'"
}

function Get-F19LiteralCount {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [string] $Literal
    )

    return [System.Text.RegularExpressions.Regex]::Matches(
        $Text,
        [System.Text.RegularExpressions.Regex]::Escape($Literal)
    ).Count
}

function Invoke-F19CapturedNative {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,

        [Parameter(Mandatory = $true)]
        [string[]] $Arguments,

        [Parameter(Mandatory = $true)]
        [string] $LogPath,

        [Parameter(Mandatory = $true)]
        [hashtable] $Environment,

        [Parameter(Mandatory = $true)]
        [string] $RepositoryRoot,

        [Parameter(Mandatory = $true)]
        [string] $IntegrityPath,

        [Parameter(Mandatory = $true)]
        [ValidatePattern('^[0-9A-Fa-f]{64}$')]
        [string] $ExpectedIntegritySha256
    )

    Assert-F19 ([System.IO.File]::Exists($FilePath)) (
        "native executable does not exist: $FilePath"
    )
    Assert-F19PathInsideRepository $LogPath $RepositoryRoot 'MATLAB log'

    $savedEnvironment = @{}
    foreach ($name in $Environment.Keys) {
        $savedEnvironment[$name] = [System.Environment]::GetEnvironmentVariable(
            [string]$name,
            [System.EnvironmentVariableTarget]::Process
        )
    }

    $hadNativePreference =
        Test-Path variable:PSNativeCommandUseErrorActionPreference
    if ($hadNativePreference) {
        $savedNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $savedErrorActionPreference = $ErrorActionPreference

    try {
        foreach ($name in $Environment.Keys) {
            [System.Environment]::SetEnvironmentVariable(
                [string]$name,
                [string]$Environment[$name],
                [System.EnvironmentVariableTarget]::Process
            )
        }
        $ErrorActionPreference = 'Continue'
        $preInvocationHash = (
            Get-FileHash $IntegrityPath -Algorithm SHA256
        ).Hash.ToUpperInvariant()
        Assert-F19 (
            $preInvocationHash -eq
                $ExpectedIntegritySha256.ToUpperInvariant()
        ) (
            "MEX hash changed immediately before MATLAB invocation: " +
            "actual=$preInvocationHash " +
            "expected=$($ExpectedIntegritySha256.ToUpperInvariant())"
        )
        $outputLines = @(& $FilePath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
        $postInvocationHash = (
            Get-FileHash $IntegrityPath -Algorithm SHA256
        ).Hash.ToUpperInvariant()
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
        foreach ($name in $savedEnvironment.Keys) {
            [System.Environment]::SetEnvironmentVariable(
                [string]$name,
                $savedEnvironment[$name],
                [System.EnvironmentVariableTarget]::Process
            )
        }
        if ($hadNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $savedNativePreference
        }
    }

    $renderedLines = @(
        foreach ($line in $outputLines) {
            $line.ToString()
        }
    )
    $text = [string]::Join("`n", $renderedLines)
    if ($text.Length -gt 0) {
        $text += "`n"
    }
    Write-F19Utf8NoBom $LogPath $text $RepositoryRoot
    if ($text.Length -gt 0) {
        Write-Host -NoNewline $text
    }
    Assert-F19 (
        $postInvocationHash -eq $ExpectedIntegritySha256.ToUpperInvariant()
    ) (
        "MEX hash changed immediately after MATLAB invocation: " +
        "actual=$postInvocationHash " +
        "expected=$($ExpectedIntegritySha256.ToUpperInvariant())"
    )

    return [pscustomobject]@{
        ExitCode = [int]$exitCode
        Text = $text
        PreInvocationMexSha256 = $preInvocationHash
        PostInvocationMexSha256 = $postInvocationHash
    }
}

$script:RepositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..')
)
$oraclePath = Join-Path (
    $script:RepositoryRoot
) 'tests\matlab\f19_problem_writeback_oracle.m'
$bindingsDirectory = Join-Path $script:RepositoryRoot 'bindings\matlab'
$testsDirectory = Join-Path $script:RepositoryRoot 'tests\matlab'

$MexBin = Get-F19FullPath $MexBin $script:RepositoryRoot
$SourceSnapshot = Get-F19FullPath (
    $SourceSnapshot
) $script:RepositoryRoot
$mexDirectory = [System.IO.Path]::GetDirectoryName($MexBin)
$RuntimeDirectory = Get-F19FullPath (
    $RuntimeDirectory
) $script:RepositoryRoot
$SummaryPath = Get-F19FullPath $SummaryPath $script:RepositoryRoot
Assert-F19PathInsideRepository (
    $RuntimeDirectory
) $script:RepositoryRoot 'runtime root'
Assert-F19PathInsideRepository (
    $SourceSnapshot
) $script:RepositoryRoot 'captured source snapshot'
Assert-F19PathInsideRepository (
    $SummaryPath
) $script:RepositoryRoot 'profile summary'

Assert-F19 ([System.IO.File]::Exists($MexBin)) (
    "supplied MEX does not exist: $MexBin"
)
Assert-F19 (
    [string]::Equals(
        [System.IO.Path]::GetFileName($MexBin),
        'dtwc_mex.mexw64',
        [System.StringComparison]::OrdinalIgnoreCase
    )
) "supplied MEX must be named dtwc_mex.mexw64: $MexBin"
Assert-F19 ([System.IO.File]::Exists($oraclePath)) (
    "MATLAB oracle does not exist: $oraclePath"
)
Assert-F19 ([System.IO.File]::Exists($SourceSnapshot)) (
    "captured MEX source does not exist: $SourceSnapshot"
)

if ([string]::IsNullOrWhiteSpace($MatlabR2024b)) {
    $MatlabR2024b = 'C:\Program Files\MATLAB\R2024b\bin\matlab.exe'
}
else {
    $MatlabR2024b = Get-F19FullPath (
        $MatlabR2024b
    ) $script:RepositoryRoot
}
if ([string]::IsNullOrWhiteSpace($MatlabR2025b)) {
    $MatlabR2025b = 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe'
}
else {
    $MatlabR2025b = Get-F19FullPath (
        $MatlabR2025b
    ) $script:RepositoryRoot
}
Assert-F19 ([System.IO.File]::Exists($MatlabR2024b)) (
    "MATLAB R2024b executable does not exist: $MatlabR2024b"
)
Assert-F19 ([System.IO.File]::Exists($MatlabR2025b)) (
    "MATLAB R2025b executable does not exist: $MatlabR2025b"
)
$MatlabR2024b = Resolve-F19ExecutableFile (
    $MatlabR2024b
) 'MATLAB R2024b executable'
$MatlabR2025b = Resolve-F19ExecutableFile (
    $MatlabR2025b
) 'MATLAB R2025b executable'
Assert-F19 (
    -not [string]::Equals(
        $MatlabR2024b,
        $MatlabR2025b,
        [System.StringComparison]::OrdinalIgnoreCase
    )
) (
    "MATLAB R2024b and R2025b resolve to the same executable: " +
    $MatlabR2024b
)

$availableReleases = @(
    [pscustomobject]@{ Name = 'R2024b'; Executable = $MatlabR2024b },
    [pscustomobject]@{ Name = 'R2025b'; Executable = $MatlabR2025b }
)
$selectedReleases = @(
    if ($Release -eq 'both') {
        $availableReleases
    }
    else {
        $availableReleases | Where-Object { $_.Name -eq $Release }
    }
)
Assert-F19 ($selectedReleases.Count -ge 1) 'no MATLAB release was selected.'
foreach ($item in $selectedReleases) {
    Assert-F19 ([System.IO.File]::Exists($item.Executable)) (
        "selected MATLAB $($item.Name) does not exist: $($item.Executable)"
    )
}

$routeNames = if ($Route -eq 'all') {
    @('fast_pam', 'fast_clara', 'clarans', 'cut_dendrogram')
}
else {
    @($Route)
}
$executionsPerVersion = $routeNames.Count
$vectorAssertionsPerVersion = (
    $executionsPerVersion * $script:MinimumVectorAssertions
)
$scalarAssertionsPerVersion = (
    $executionsPerVersion * $script:ScalarAssertionsPerExecution
)
$mexHash = (
    Get-FileHash $MexBin -Algorithm SHA256
).Hash.ToUpperInvariant()
$mexSourceHash = (
    Get-FileHash $SourceSnapshot -Algorithm SHA256
).Hash.ToUpperInvariant()
$ExpectedMexSha256 = $ExpectedMexSha256.ToUpperInvariant()
$ExpectedSourceSha256 = $ExpectedSourceSha256.ToUpperInvariant()
Assert-F19 (
    $mexHash -eq $ExpectedMexSha256
) (
    "MEX SHA-256 mismatch: actual=$mexHash " +
    "expected=$ExpectedMexSha256"
)
Assert-F19 (
    $mexSourceHash -eq $ExpectedSourceSha256
) (
    "captured source SHA-256 mismatch: actual=$mexSourceHash " +
    "expected=$ExpectedSourceSha256"
)

New-F19Directory $RuntimeDirectory $script:RepositoryRoot
$repoLiteral = ConvertTo-F19MatlabLiteral $script:RepositoryRoot
$bindingsLiteral = ConvertTo-F19MatlabLiteral $bindingsDirectory
$testsLiteral = ConvertTo-F19MatlabLiteral $testsDirectory
$mexDirectoryLiteral = ConvertTo-F19MatlabLiteral $mexDirectory
$mexPathLiteral = ConvertTo-F19MatlabLiteral $MexBin
$routeLiteral = ConvertTo-F19MatlabLiteral $Route
$profileLiteral = ConvertTo-F19MatlabLiteral $Profile

$totalExecutions = 0
$totalVectorAssertions = 0
$totalScalarAssertions = 0
$releaseRecords = @()
$observedReleases = @()

foreach ($item in $selectedReleases) {
    $version = $item.Name
    $versionLiteral = ConvertTo-F19MatlabLiteral $version
    $runDirectory = Join-Path (
        $RuntimeDirectory
    ) (Join-Path $Profile (Join-Path $Route $version))
    $preferences = Join-Path $runDirectory 'matlab-pref'
    $temporary = Join-Path $runDirectory 'temp'
    $driverPath = Join-Path $runDirectory 'f19_writeback_driver.m'
    $logPath = Join-Path $runDirectory 'matlab.log'
    New-F19Directory $preferences $script:RepositoryRoot
    New-F19Directory $temporary $script:RepositoryRoot

    $driver = @"
restoredefaultpath;
observedRelease=['R' char(version('-release'))];
if ~strcmp(observedRelease,$versionLiteral)
    error('dtwc:f19MatlabRelease', ...
        'MATLAB release mismatch: requested=%s observed=%s.', ...
        $versionLiteral,observedRelease);
end
fprintf(['F19_MATLAB_RELEASE requested_release=$version ' ...
    'observed_release=%s\n'],observedRelease);
cd($repoLiteral);
addpath($bindingsLiteral);
addpath($testsLiteral);
addpath($mexDirectoryLiteral);
clear dtwc_mex;
rehash;
mexPaths=which('dtwc_mex','-all');
if ischar(mexPaths)
    if isempty(mexPaths), mexPaths={}; else, mexPaths={mexPaths}; end
elseif isstring(mexPaths)
    mexPaths=cellstr(mexPaths);
end
for i=1:numel(mexPaths)
    fprintf(['F19_MEX_ALL requested_release=$version ' ...
        'observed_release=%s profile=$Profile index=%d path=%s\n'], ...
        observedRelease,i,mexPaths{i});
end
if numel(mexPaths)~=1
    error('dtwc:f19MexPath','Expected exactly one dtwc_mex; observed %d.',numel(mexPaths));
end
resolvedMex=strrep(mexPaths{1},char(92),'/');
if ~strcmp(resolvedMex,$mexPathLiteral)
    error('dtwc:f19MexPath','Resolved MEX mismatch: actual=%s expected=%s.',resolvedMex,$mexPathLiteral);
end
fprintf(['F19_MEX_PATH requested_release=$version observed_release=%s ' ...
    'profile=$Profile path=%s mex_sha256=$mexHash ' ...
    'source_sha256=$mexSourceHash\n'],observedRelease,mexPaths{1});
summary=f19_problem_writeback_oracle( ...
    $routeLiteral,$profileLiteral,observedRelease);
expectedExecutions=$executionsPerVersion;
expectedVectorAssertions=$vectorAssertionsPerVersion;
expectedScalarAssertions=$scalarAssertionsPerVersion;
assert(summary.executions==expectedExecutions);
assert(summary.vector_assertions==expectedVectorAssertions);
assert(summary.minimum_vector_assertions==$($script:MinimumVectorAssertions));
assert(summary.scalar_assertions==expectedScalarAssertions);
fprintf(['F19_MATLAB_PROFILE profile=%s requested_release=$version ' ...
    'observed_release=%s requested_route=%s routes=%d/%d ' ...
    'executions=%d vector_assertions=%d/%d ' ...
    'minimum_vector_assertions=%d scalar_assertions=%d/%d skips=0\n'], ...
    $profileLiteral,observedRelease,$routeLiteral, ...
    summary.executions,expectedExecutions,summary.executions, ...
    summary.vector_assertions,expectedVectorAssertions, ...
    summary.minimum_vector_assertions,summary.scalar_assertions, ...
    expectedScalarAssertions);
"@
    Write-F19Utf8NoBom $driverPath $driver $script:RepositoryRoot

    $runExpression = 'run(' + (
        ConvertTo-F19MatlabLiteral $driverPath
    ) + ');'
    $pathWithMex = $mexDirectory + [System.IO.Path]::PathSeparator + $env:PATH
    $environment = @{
        'PATH' = $pathWithMex
        'MATLAB_PREFDIR' = $preferences
        'TEMP' = $temporary
        'TMP' = $temporary
    }
    $result = Invoke-F19CapturedNative $item.Executable @(
        '-batch',
        $runExpression
    ) $logPath $environment $script:RepositoryRoot (
        $MexBin
    ) $ExpectedMexSha256

    Assert-F19 ($result.ExitCode -eq 0) (
        "MATLAB $version exited $($result.ExitCode); see $logPath"
    )
    $skipPattern = (
        '(?im)\b(?:SKIP|SKIPPED|SKIPPING|INCOMPLETE)\b|' +
        'assumption[ _-]*(?:failed|failure)|capability[ _-]*unavailable'
    )
    $skipMatches = [System.Text.RegularExpressions.Regex]::Matches(
        $result.Text,
        $skipPattern
    )
    Assert-F19 ($skipMatches.Count -eq 0) (
        "MATLAB $version emitted generic skip/incomplete text: " +
        [string]::Join(
            ' | ',
            @($skipMatches | ForEach-Object { $_.Value })
        )
    )
    $releasePattern = (
        '(?m)^F19_MATLAB_RELEASE requested_release=' +
        [System.Text.RegularExpressions.Regex]::Escape($version) +
        ' observed_release=(?<observed>R[0-9]{4}[ab])\r?$'
    )
    $releaseMatches = [System.Text.RegularExpressions.Regex]::Matches(
        $result.Text,
        $releasePattern
    )
    Assert-F19 (
        $releaseMatches.Count -eq 1
    ) "MATLAB $version did not print exactly one observed-release marker."
    $observedRelease = $releaseMatches[0].Groups['observed'].Value
    Assert-F19 (
        $observedRelease -eq $version
    ) (
        "MATLAB release identity drift: requested=$version " +
        "observed=$observedRelease"
    )
    $observedReleases += $observedRelease

    $mexPathMarker = (
        "F19_MEX_PATH requested_release=$version " +
        "observed_release=$observedRelease profile=$Profile "
    )
    Assert-F19 (
        (Get-F19LiteralCount $result.Text $mexPathMarker) -eq 1
    ) "MATLAB $version did not print exactly one MEX path marker."

    $routeMarkerPrefix = (
        "F19_MATLAB_ROUTE profile=$Profile version=$observedRelease " +
        "observed_release=$observedRelease route="
    )
    Assert-F19 (
        (Get-F19LiteralCount $result.Text $routeMarkerPrefix) -eq
            $executionsPerVersion
    ) (
        "MATLAB $version route-marker count drift: expected " +
        "$executionsPerVersion."
    )
    foreach ($routeName in $routeNames) {
        $routeMarker = (
            "F19_MATLAB_ROUTE profile=$Profile " +
            "version=$observedRelease " +
            "observed_release=$observedRelease " +
            "route=$routeName executions=1 " +
            "vector_assertions=$($script:MinimumVectorAssertions)/" +
            "$($script:MinimumVectorAssertions) " +
            "minimum_vector_assertions=$($script:MinimumVectorAssertions) " +
            "scalar_assertions=$($script:ScalarAssertionsPerExecution)/" +
            "$($script:ScalarAssertionsPerExecution)"
        )
        Assert-F19 (
            (Get-F19LiteralCount $result.Text $routeMarker) -eq 1
        ) "MATLAB $version route '$routeName' marker is absent or duplicated."
    }

    $profileMarker = (
        "F19_MATLAB_PROFILE profile=$Profile " +
        "requested_release=$version " +
        "observed_release=$observedRelease " +
        "requested_route=$Route routes=$executionsPerVersion/" +
        "$executionsPerVersion executions=$executionsPerVersion " +
        "vector_assertions=$vectorAssertionsPerVersion/" +
        "$vectorAssertionsPerVersion " +
        "minimum_vector_assertions=$($script:MinimumVectorAssertions) " +
        "scalar_assertions=$scalarAssertionsPerVersion/" +
        "$scalarAssertionsPerVersion skips=0"
    )
    Assert-F19 (
        (Get-F19LiteralCount $result.Text $profileMarker) -eq 1
    ) "MATLAB $version did not print its exact profile marker once."
    Write-Host (
        "F19_MATLAB_INTEGRITY requested_release=$version " +
        "observed_release=$observedRelease " +
        "mex_sha256_before=$($result.PreInvocationMexSha256) " +
        "mex_sha256_after=$($result.PostInvocationMexSha256) verdict=PASS"
    )

    $totalExecutions += $executionsPerVersion
    $totalVectorAssertions += $vectorAssertionsPerVersion
    $totalScalarAssertions += $scalarAssertionsPerVersion
    $releaseRecords += [ordered]@{
        requested_release = $version
        observed_release = $observedRelease
        matlab_executable = $item.Executable
        mex_sha256_before = $result.PreInvocationMexSha256
        mex_sha256_after = $result.PostInvocationMexSha256
        log = $logPath
        log_sha256 = (
            Get-FileHash $logPath -Algorithm SHA256
        ).Hash.ToUpperInvariant()
        executions = $executionsPerVersion
        vector_assertions = $vectorAssertionsPerVersion
        scalar_assertions = $scalarAssertionsPerVersion
        route_markers = $executionsPerVersion
        skips = 0
    }
}

$expectedVersions = $selectedReleases.Count
$expectedTotalExecutions = $executionsPerVersion * $expectedVersions
$expectedTotalVectors = $vectorAssertionsPerVersion * $expectedVersions
$expectedTotalScalars = $scalarAssertionsPerVersion * $expectedVersions
Assert-F19 (
    $totalExecutions -eq $expectedTotalExecutions
) 'aggregate execution count drift.'
Assert-F19 (
    $totalVectorAssertions -eq $expectedTotalVectors
) 'aggregate vector-assertion count drift.'
Assert-F19 (
    $totalScalarAssertions -eq $expectedTotalScalars
) 'aggregate scalar-assertion count drift.'

$summary = [ordered]@{
    schema = 'dtwc.f19.matlab-profile-run.v1'
    profile = $Profile
    requested_route = $Route
    requested_releases = @(
        $selectedReleases | ForEach-Object { $_.Name }
    )
    observed_releases = $observedReleases
    versions = $expectedVersions
    release_identity_checks = $expectedVersions
    routes_per_version = $executionsPerVersion
    executions = $totalExecutions
    vector_assertions = $totalVectorAssertions
    minimum_vector_assertions_per_execution =
        $script:MinimumVectorAssertions
    scalar_assertions = $totalScalarAssertions
    scalar_assertions_per_execution =
        $script:ScalarAssertionsPerExecution
    route_markers = $totalExecutions
    mex_path = $MexBin
    mex_sha256 = $mexHash
    mex_hash_checks = $expectedVersions * 2
    source_snapshot = $SourceSnapshot
    source_sha256 = $mexSourceHash
    skips = 0
    profile_verdict = 'FINAL'
    release_records = $releaseRecords
}
$summaryText = $summary | ConvertTo-Json -Depth 8
Write-F19Utf8NoBom (
    $SummaryPath
) ($summaryText + "`n") $script:RepositoryRoot

Write-Host (
    "F19_MATLAB_PROFILE_FINAL profile=$Profile requested_route=$Route " +
    "versions=$expectedVersions/$expectedVersions " +
    "release_identity_checks=$expectedVersions/$expectedVersions " +
    "observed_releases=$([string]::Join(',', $observedReleases)) " +
    "routes_per_version=$executionsPerVersion/$executionsPerVersion " +
    "executions=$totalExecutions/$expectedTotalExecutions " +
    "vector_assertions=$totalVectorAssertions/$expectedTotalVectors " +
    "minimum_vector_assertions_per_execution=" +
    "$($script:MinimumVectorAssertions) " +
    "scalar_assertions=$totalScalarAssertions/$expectedTotalScalars " +
    "mex_hash_checks=$($expectedVersions * 2)/" +
    "$($expectedVersions * 2) " +
    "mex_sha256=$mexHash source_sha256=$mexSourceHash skips=0 " +
    "profile_verdict=FINAL summary=$SummaryPath"
)
