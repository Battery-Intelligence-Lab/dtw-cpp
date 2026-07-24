[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        'inherited',
        'single_fast_pam',
        'single_fast_clara',
        'single_clarans',
        'single_cut_dendrogram',
        'composite'
    )]
    [string] $Profile,

    [Parameter(Mandatory = $true)]
    [string] $BuildDirectory,

    [ValidateSet('Release', 'RelWithDebInfo', 'Debug', 'MinSizeRel')]
    [string] $Configuration = 'Release',

    [string] $EvidenceRoot = 'build\f19-matlab-writeback\profiles',

    [string] $CMake = 'cmake'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'f19_matlab_writeback_evidence.ps1')

function Invoke-F19CaptureBuild {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,

        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    $hadNativePreference =
        Test-Path variable:PSNativeCommandUseErrorActionPreference
    if ($hadNativePreference) {
        $savedNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $savedErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $outputLines = @(& $FilePath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
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
        Write-Host -NoNewline $text
    }
    return [pscustomobject]@{
        ExitCode = [int]$exitCode
        Text = $text
    }
}

$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..')
)
$sourcePath = Join-Path $repositoryRoot 'bindings\matlab\dtwc_mex.cpp'
$BuildDirectory = Resolve-F19RepositoryPath (
    $BuildDirectory
) $repositoryRoot
$EvidenceRoot = Resolve-F19RepositoryPath $EvidenceRoot $repositoryRoot
Assert-F19EvidencePathInside (
    $BuildDirectory
) $repositoryRoot 'MEX build directory'
Assert-F19EvidencePathInside (
    $EvidenceRoot
) $repositoryRoot 'profile evidence root'

$cachePath = Join-Path $BuildDirectory 'CMakeCache.txt'
Assert-F19Evidence (
    [System.IO.File]::Exists($cachePath)
) "configured CMake cache does not exist: $cachePath"
$cacheText = [System.IO.File]::ReadAllText($cachePath)
$homeMatch = [System.Text.RegularExpressions.Regex]::Match(
    $cacheText,
    '(?m)^CMAKE_HOME_DIRECTORY:INTERNAL=(.+?)\r?$'
)
Assert-F19Evidence (
    $homeMatch.Success
) "CMake cache does not identify CMAKE_HOME_DIRECTORY: $cachePath"
$configuredHome = [System.IO.Path]::GetFullPath(
    $homeMatch.Groups[1].Value.Trim()
)
Assert-F19Evidence (
    [string]::Equals(
        $configuredHome,
        $repositoryRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )
) (
    "CMake build belongs to '$configuredHome', not '$repositoryRoot'."
)
Assert-F19Evidence (
    [System.Text.RegularExpressions.Regex]::IsMatch(
        $cacheText,
        '(?m)^DTWC_BUILD_MATLAB:BOOL=ON\r?$'
    )
) "CMake build does not have DTWC_BUILD_MATLAB=ON: $cachePath"

$MexBin = [System.IO.Path]::GetFullPath(
    (Join-Path $BuildDirectory 'bin\dtwc_mex.mexw64')
)
Assert-F19EvidencePathInside (
    $MexBin
) $BuildDirectory 'MEX output'
Assert-F19Evidence (
    [string]::Equals(
        [System.IO.Path]::GetFileName($MexBin),
        'dtwc_mex.mexw64',
        [System.StringComparison]::OrdinalIgnoreCase
    )
) "MEX output must be named dtwc_mex.mexw64: $MexBin"

$profileDirectory = Join-Path $EvidenceRoot $Profile
Assert-F19Evidence (
    -not [System.IO.Directory]::Exists($profileDirectory)
) (
    "profile evidence already exists and will not be overwritten: " +
    $profileDirectory
)

$specification = Get-F19ProfileSpecification $Profile
$sourceEvidenceBefore = Read-F19StrictUtf8NoBom (
    $sourcePath
) 'tracked MEX source'
$sourceTextBefore = $sourceEvidenceBefore.Text
$sourceShape = Assert-F19ProfileSource $Profile $sourceTextBefore
$sourceHashBefore = Get-F19Sha256 $sourcePath

if (
    $Profile.StartsWith('single_', [System.StringComparison]::Ordinal) -or
    $Profile -eq 'composite'
) {
    $inheritedSourcePath = Join-Path (
        (Join-Path $EvidenceRoot 'inherited')
    ) 'dtwc_mex.cpp'
    Assert-F19Evidence (
        [System.IO.File]::Exists($inheritedSourcePath)
    ) (
        "capture inherited before '$Profile'; snapshot is absent: " +
        $inheritedSourcePath
    )
    $inheritedEvidence = Read-F19StrictUtf8NoBom (
        $inheritedSourcePath
    ) 'captured inherited source'
    $inheritedText = $inheritedEvidence.Text
}

if ($Profile.StartsWith('single_', [System.StringComparison]::Ordinal)) {
    $deletedRoute = $Profile.Substring('single_'.Length)
    $expectedText = Remove-F19RouteCallLine $inheritedText $deletedRoute
    $expectedBytes = ConvertTo-F19StrictUtf8NoBomBytes $expectedText
    Assert-F19Evidence (
        Test-F19ByteEquality $sourceEvidenceBefore.Bytes $expectedBytes
    ) (
        "profile '$Profile' is not the inherited snapshot with only the " +
        "'$deletedRoute' helper-call line deleted, byte-for-byte."
    )
}
elseif ($Profile -eq 'composite') {
    $expectedText = Get-F19CompositeSource $inheritedText
    $expectedBytes = ConvertTo-F19StrictUtf8NoBomBytes $expectedText
    Assert-F19Evidence (
        Test-F19ByteEquality $sourceEvidenceBefore.Bytes $expectedBytes
    ) (
        "profile 'composite' is not byte-exact inherited source with only " +
        'the helper definition and four helper-call lines deleted.'
    )
}

$cmakeCommand = Get-Command $CMake -CommandType Application -ErrorAction Stop |
    Select-Object -First 1
$cmakePath = $cmakeCommand.Source
$buildArguments = @(
    '--build',
    $BuildDirectory,
    '--target',
    'dtwc_mex',
    '--config',
    $Configuration,
    '--clean-first'
)
$buildStartUtc = [System.DateTime]::UtcNow
$buildResult = Invoke-F19CaptureBuild $cmakePath $buildArguments
Assert-F19Evidence (
    $buildResult.ExitCode -eq 0
) (
    "clean target rebuild for profile '$Profile' exited " +
    "$($buildResult.ExitCode)."
)

$sourceHashAfter = Get-F19Sha256 $sourcePath
Assert-F19Evidence (
    $sourceHashAfter -eq $sourceHashBefore
) (
    "tracked MEX source changed during profile '$Profile' build: " +
    "before=$sourceHashBefore after=$sourceHashAfter."
)
$sourceEvidenceAfter = Read-F19StrictUtf8NoBom (
    $sourcePath
) 'tracked MEX source after build'
Assert-F19Evidence (
    Test-F19ByteEquality (
        $sourceEvidenceBefore.Bytes
    ) $sourceEvidenceAfter.Bytes
) "tracked MEX source bytes changed during profile '$Profile' build."
Assert-F19Evidence (
    [System.IO.File]::Exists($MexBin)
) "clean rebuild did not produce the expected MEX: $MexBin"
$mexInfo = Get-Item -LiteralPath $MexBin
Assert-F19Evidence (
    $mexInfo.Length -gt 0
) "clean rebuild produced an empty MEX: $MexBin"
Assert-F19Evidence (
    $mexInfo.LastWriteTimeUtc -ge $buildStartUtc.AddSeconds(-2)
) (
    "MEX timestamp predates the clean rebuild: MEX=" +
    "$($mexInfo.LastWriteTimeUtc.ToString('o')) build_start=" +
    "$($buildStartUtc.ToString('o'))."
)

$captureMarker = (
    "F19_CAPTURE_BUILD profile=$Profile " +
    "source_sha256=$sourceHashBefore clean_first=1 " +
    "expected_mex_path=$($MexBin.Replace('\', '/'))"
)
$quotedArguments = @(
    foreach ($argument in $buildArguments) {
        '"' + $argument.Replace('"', '""') + '"'
    }
)
$commandMarker = (
    'F19_CAPTURE_COMMAND executable="' +
    $cmakePath.Replace('"', '""') +
    '" arguments=' +
    [string]::Join(' ', $quotedArguments)
)
$buildLogText = (
    $captureMarker + "`n" +
    $commandMarker + "`n" +
    $buildResult.Text
)
$buildAudit = Assert-F19BuildLog (
    $buildLogText
) $Profile $sourceHashBefore $MexBin
$mexHash = Get-F19Sha256 $MexBin

foreach ($other in Get-F19ProfileSpecifications) {
    if ($other.Name -eq $Profile) {
        continue
    }
    $otherDirectory = Join-Path $EvidenceRoot $other.Name
    $otherSource = Join-Path $otherDirectory 'dtwc_mex.cpp'
    $otherMex = Join-Path $otherDirectory 'dtwc_mex.mexw64'
    if ([System.IO.File]::Exists($otherSource)) {
        Assert-F19Evidence (
            (Get-F19Sha256 $otherSource) -ne $sourceHashBefore
        ) (
            "profile '$Profile' source duplicates captured profile " +
            "'$($other.Name)'."
        )
    }
    if ([System.IO.File]::Exists($otherMex)) {
        Assert-F19Evidence (
            (Get-F19Sha256 $otherMex) -ne $mexHash
        ) (
            "profile '$Profile' MEX duplicates captured profile " +
            "'$($other.Name)'."
        )
    }
}

if (-not [System.IO.Directory]::Exists($EvidenceRoot)) {
    [void][System.IO.Directory]::CreateDirectory($EvidenceRoot)
}
[void][System.IO.Directory]::CreateDirectory($profileDirectory)
$capturedSourcePath = Join-Path $profileDirectory 'dtwc_mex.cpp'
$capturedMexPath = Join-Path $profileDirectory 'dtwc_mex.mexw64'
$capturedBuildLogPath = Join-Path $profileDirectory 'build.log'
[System.IO.File]::Copy($sourcePath, $capturedSourcePath, $false)
[System.IO.File]::Copy($MexBin, $capturedMexPath, $false)
Write-F19EvidenceUtf8NoBom (
    $capturedBuildLogPath
) $buildLogText $repositoryRoot

Assert-F19Evidence (
    (Get-F19Sha256 $capturedSourcePath) -eq $sourceHashBefore
) "captured source bytes do not match the built source."
$capturedSourceEvidence = Read-F19StrictUtf8NoBom (
    $capturedSourcePath
) "captured profile '$Profile' source"
Assert-F19Evidence (
    Test-F19ByteEquality (
        $sourceEvidenceBefore.Bytes
    ) $capturedSourceEvidence.Bytes
) "captured source is not byte-identical to the built source."
Assert-F19Evidence (
    (Get-F19Sha256 $capturedMexPath) -eq $mexHash
) "captured MEX bytes do not match the fresh build output."
$buildLogHash = Get-F19Sha256 $capturedBuildLogPath

$callManifest = [ordered]@{}
foreach ($route in $sourceShape.Calls.Keys) {
    $callManifest[$route] = $sourceShape.Calls[$route]
}
$manifest = [ordered]@{
    schema = 'dtwc.f19.matlab-profile.v1'
    profile = $Profile
    route = $specification.Route
    expected_executions = $specification.Executions
    captured_at_utc = [System.DateTime]::UtcNow.ToString('o')
    source_file = 'dtwc_mex.cpp'
    source_sha256 = $sourceHashBefore
    source_encoding = 'UTF-8'
    source_bom = $false
    mex_file = 'dtwc_mex.mexw64'
    mex_sha256 = $mexHash
    mex_size_bytes = [int64]$mexInfo.Length
    build_log_file = 'build.log'
    build_log_sha256 = $buildLogHash
    build_directory = $BuildDirectory
    build_mex_path = $MexBin
    cmake_home_directory = $configuredHome
    cmake_executable = $cmakePath
    build_configuration = $Configuration
    clean_first = $true
    compile_lines = $buildAudit.CompileLines
    link_lines = $buildAudit.LinkLines
    bound_link_lines = $buildAudit.BoundLinkLines
    semantics = [ordered]@{
        helper_definitions = $sourceShape.HelperDefinitions
        helper_set_k = $sourceShape.HelperSetK
        helper_set_medoids = $sourceShape.HelperSetMedoids
        helper_set_labels = $sourceShape.HelperSetLabels
        helper_tokens = $sourceShape.Tokens
        calls = $callManifest
    }
}
$manifestPath = Join-Path $profileDirectory 'manifest.json'
$manifestText = $manifest | ConvertTo-Json -Depth 8
Write-F19EvidenceUtf8NoBom (
    $manifestPath
) ($manifestText + "`n") $repositoryRoot

Write-Host (
    "F19_MATLAB_PROFILE_CAPTURE profile=$Profile " +
    "route=$($specification.Route) " +
    "source_sha256=$sourceHashBefore mex_sha256=$mexHash " +
    "source_distinct=1 mex_distinct=1 " +
    "compile_lines=$($buildAudit.CompileLines) " +
    "link_lines=$($buildAudit.LinkLines) clean_first=1 " +
    "manifest=$manifestPath verdict=CAPTURED"
)
