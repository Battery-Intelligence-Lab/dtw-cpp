[CmdletBinding()]
param(
    [string] $EvidenceRoot = 'build\f19-matlab-writeback\profiles',
    [string] $RuntimeRoot = 'build\f19-matlab-writeback\runs',
    [string] $MatlabR2024b = '',
    [string] $MatlabR2025b = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'f19_matlab_writeback_evidence.ps1')

$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..')
)
$sourcePath = Join-Path $repositoryRoot 'bindings\matlab\dtwc_mex.cpp'
$profileRunner = Join-Path $PSScriptRoot 'test_f19_matlab_writeback.ps1'
$EvidenceRoot = Resolve-F19RepositoryPath $EvidenceRoot $repositoryRoot
$RuntimeRoot = Resolve-F19RepositoryPath $RuntimeRoot $repositoryRoot
Assert-F19EvidencePathInside (
    $EvidenceRoot
) $repositoryRoot 'profile evidence root'
Assert-F19EvidencePathInside (
    $RuntimeRoot
) $repositoryRoot 'schedule runtime root'
Assert-F19Evidence (
    [System.IO.Directory]::Exists($EvidenceRoot)
) "profile evidence root does not exist: $EvidenceRoot"
Assert-F19Evidence (
    [System.IO.File]::Exists($profileRunner)
) "per-profile runner does not exist: $profileRunner"

if ([string]::IsNullOrWhiteSpace($MatlabR2024b)) {
    $MatlabR2024b = 'C:\Program Files\MATLAB\R2024b\bin\matlab.exe'
}
else {
    $MatlabR2024b = Resolve-F19RepositoryPath (
        $MatlabR2024b
    ) $repositoryRoot
}
if ([string]::IsNullOrWhiteSpace($MatlabR2025b)) {
    $MatlabR2025b = 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe'
}
else {
    $MatlabR2025b = Resolve-F19RepositoryPath (
        $MatlabR2025b
    ) $repositoryRoot
}
Assert-F19Evidence (
    [System.IO.File]::Exists($MatlabR2024b)
) "MATLAB R2024b executable does not exist: $MatlabR2024b"
Assert-F19Evidence (
    [System.IO.File]::Exists($MatlabR2025b)
) "MATLAB R2025b executable does not exist: $MatlabR2025b"
$MatlabR2024b = Resolve-F19ExecutableFile (
    $MatlabR2024b
) 'MATLAB R2024b executable'
$MatlabR2025b = Resolve-F19ExecutableFile (
    $MatlabR2025b
) 'MATLAB R2025b executable'
Assert-F19Evidence (
    -not [string]::Equals(
        $MatlabR2024b,
        $MatlabR2025b,
        [System.StringComparison]::OrdinalIgnoreCase
    )
) (
    'MATLAB R2024b and R2025b resolve to the same executable: ' +
    $MatlabR2024b
)

$specifications = @(Get-F19ProfileSpecifications)
$expectedProfileNames = @($specifications | ForEach-Object { $_.Name })
$actualProfileNames = @(
    Get-ChildItem -LiteralPath $EvidenceRoot -Directory |
        Sort-Object Name |
        ForEach-Object { $_.Name }
)
Assert-F19Evidence (
    $actualProfileNames.Count -eq $expectedProfileNames.Count
) (
    "profile-directory count drift: observed $($actualProfileNames.Count), " +
    "expected $($expectedProfileNames.Count). Actual=[" +
    [string]::Join(',', $actualProfileNames) + '].'
)
foreach ($name in $expectedProfileNames) {
    Assert-F19Evidence (
        $actualProfileNames -contains $name
    ) "fixed profile directory '$name' is absent."
}
foreach ($name in $actualProfileNames) {
    Assert-F19Evidence (
        $expectedProfileNames -contains $name
    ) "unexpected profile directory '$name' is present."
}

$evidenceByProfile = [ordered]@{}
$sourceHashes = @()
$mexHashes = @()
$compileLogCount = 0
$semanticProfileCount = 0

foreach ($specification in $specifications) {
    $profile = $specification.Name
    $profileDirectory = Join-Path $EvidenceRoot $profile
    $actualFiles = @(
        Get-ChildItem -LiteralPath $profileDirectory -File |
            Sort-Object Name |
            ForEach-Object { $_.Name }
    )
    $expectedFiles = @(
        'build.log',
        'dtwc_mex.cpp',
        'dtwc_mex.mexw64',
        'manifest.json'
    )
    Assert-F19Evidence (
        $actualFiles.Count -eq $expectedFiles.Count
    ) (
        "profile '$profile' file-count drift: observed " +
        "$($actualFiles.Count), expected $($expectedFiles.Count). Actual=[" +
        [string]::Join(',', $actualFiles) + '].'
    )
    foreach ($file in $expectedFiles) {
        Assert-F19Evidence (
            $actualFiles -contains $file
        ) "profile '$profile' is missing '$file'."
    }

    $sourceSnapshot = Join-Path $profileDirectory 'dtwc_mex.cpp'
    $mexPath = Join-Path $profileDirectory 'dtwc_mex.mexw64'
    $buildLogPath = Join-Path $profileDirectory 'build.log'
    $manifestPath = Join-Path $profileDirectory 'manifest.json'
    $manifest = (
        [System.IO.File]::ReadAllText($manifestPath) |
            ConvertFrom-Json
    )
    Assert-F19Evidence (
        $manifest.schema -eq 'dtwc.f19.matlab-profile.v1'
    ) "profile '$profile' manifest schema drift."
    Assert-F19Evidence (
        $manifest.profile -eq $profile
    ) "profile '$profile' manifest identity drift."
    Assert-F19Evidence (
        $manifest.route -eq $specification.Route
    ) (
        "profile '$profile' manifest route '$($manifest.route)' does not " +
        "match '$($specification.Route)'."
    )
    Assert-F19Evidence (
        [int]$manifest.expected_executions -eq $specification.Executions
    ) "profile '$profile' manifest execution band drift."
    Assert-F19Evidence (
        $manifest.source_file -eq 'dtwc_mex.cpp'
    ) "profile '$profile' manifest source filename drift."
    Assert-F19Evidence (
        $manifest.mex_file -eq 'dtwc_mex.mexw64'
    ) "profile '$profile' manifest MEX filename drift."
    Assert-F19Evidence (
        $manifest.build_log_file -eq 'build.log'
    ) "profile '$profile' manifest build-log filename drift."
    Assert-F19Evidence (
        $manifest.source_encoding -eq 'UTF-8' -and
        -not [bool]$manifest.source_bom
    ) "profile '$profile' manifest source encoding/BOM contract drift."
    Assert-F19Evidence (
        [bool]$manifest.clean_first
    ) "profile '$profile' was not captured with clean_first=true."
    Assert-F19Evidence (
        [string]::Equals(
            [System.IO.Path]::GetFullPath(
                [string]$manifest.cmake_home_directory
            ),
            $repositoryRoot,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) "profile '$profile' CMake home is not this repository."
    $manifestBuildDirectory = [System.IO.Path]::GetFullPath(
        [string]$manifest.build_directory
    )
    Assert-F19EvidencePathInside (
        $manifestBuildDirectory
    ) $repositoryRoot "profile '$profile' build directory"
    $expectedBuildMex = [System.IO.Path]::GetFullPath(
        (Join-Path $manifestBuildDirectory 'bin\dtwc_mex.mexw64')
    )
    Assert-F19Evidence (
        [string]::Equals(
            [System.IO.Path]::GetFullPath(
                [string]$manifest.build_mex_path
            ),
            $expectedBuildMex,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) "profile '$profile' manifest does not bind the exact build MEX path."

    $sourceHash = Get-F19Sha256 $sourceSnapshot
    $mexHash = Get-F19Sha256 $mexPath
    $buildLogHash = Get-F19Sha256 $buildLogPath
    Assert-F19Evidence (
        $sourceHash -eq ([string]$manifest.source_sha256).ToUpperInvariant()
    ) "profile '$profile' captured source hash does not match its manifest."
    Assert-F19Evidence (
        $mexHash -eq ([string]$manifest.mex_sha256).ToUpperInvariant()
    ) "profile '$profile' captured MEX hash does not match its manifest."
    Assert-F19Evidence (
        $buildLogHash -eq (
            [string]$manifest.build_log_sha256
        ).ToUpperInvariant()
    ) "profile '$profile' build-log hash does not match its manifest."
    Assert-F19Evidence (
        (Get-Item -LiteralPath $mexPath).Length -eq
            [int64]$manifest.mex_size_bytes
    ) "profile '$profile' captured MEX size does not match its manifest."

    $sourceEvidence = Read-F19StrictUtf8NoBom (
        $sourceSnapshot
    ) "profile '$profile' captured source"
    $sourceText = $sourceEvidence.Text
    $sourceShape = Assert-F19ProfileSource $profile $sourceText
    $semanticProfileCount += 1
    $buildLogText = [System.IO.File]::ReadAllText($buildLogPath)
    $buildAudit = Assert-F19BuildLog (
        $buildLogText
    ) $profile $sourceHash $expectedBuildMex
    Assert-F19Evidence (
        [int]$manifest.compile_lines -eq $buildAudit.CompileLines
    ) "profile '$profile' compile-line count does not match its manifest."
    Assert-F19Evidence (
        [int]$manifest.link_lines -eq $buildAudit.LinkLines
    ) "profile '$profile' link-line count does not match its manifest."
    Assert-F19Evidence (
        [int]$manifest.bound_link_lines -eq $buildAudit.BoundLinkLines -and
        $buildAudit.BoundLinkLines -ge 1
    ) "profile '$profile' exact build-MEX link binding drift."
    $compileLogCount += 1

    $sourceHashes += $sourceHash
    $mexHashes += $mexHash
    $evidenceByProfile[$profile] = [pscustomobject]@{
        Specification = $specification
        Directory = $profileDirectory
        Source = $sourceSnapshot
        SourceText = $sourceText
        SourceBytes = $sourceEvidence.Bytes
        SourceHash = $sourceHash
        SourceShape = $sourceShape
        Mex = $mexPath
        MexHash = $mexHash
        BuildLog = $buildLogPath
        BuildLogHash = $buildLogHash
        ManifestPath = $manifestPath
        ManifestHash = Get-F19Sha256 $manifestPath
        ExpectedBuildMex = $expectedBuildMex
        Manifest = $manifest
    }
}

$inheritedText = $evidenceByProfile['inherited'].SourceText
foreach ($route in @(
    'fast_pam',
    'fast_clara',
    'clarans',
    'cut_dendrogram'
)) {
    $profile = "single_$route"
    $expectedSingleText = Remove-F19RouteCallLine $inheritedText $route
    $expectedSingleBytes = ConvertTo-F19StrictUtf8NoBomBytes (
        $expectedSingleText
    )
    Assert-F19Evidence (
        Test-F19ByteEquality (
            $evidenceByProfile[$profile].SourceBytes
        ) $expectedSingleBytes
    ) (
        "profile '$profile' is not the inherited source with exactly the " +
        "'$route' helper-call line deleted, byte-for-byte."
    )
}
$expectedCompositeText = Get-F19CompositeSource $inheritedText
$expectedCompositeBytes = ConvertTo-F19StrictUtf8NoBomBytes (
    $expectedCompositeText
)
Assert-F19Evidence (
    Test-F19ByteEquality (
        $evidenceByProfile['composite'].SourceBytes
    ) $expectedCompositeBytes
) (
    "profile 'composite' is not byte-exact inherited source with only the " +
    'helper definition and four helper-call lines deleted.'
)

$uniqueSourceHashes = @($sourceHashes | Sort-Object -Unique)
$uniqueMexHashes = @($mexHashes | Sort-Object -Unique)
Assert-F19Evidence (
    $uniqueSourceHashes.Count -eq $specifications.Count
) (
    "source artifacts are not six-way distinct: observed " +
    "$($uniqueSourceHashes.Count), expected $($specifications.Count)."
)
Assert-F19Evidence (
    $uniqueMexHashes.Count -eq $specifications.Count
) (
    "MEX artifacts are not six-way distinct: observed " +
    "$($uniqueMexHashes.Count), expected $($specifications.Count)."
)

$liveSourceEvidence = Read-F19StrictUtf8NoBom (
    $sourcePath
) 'live composite MEX source'
$liveSourceHash = Get-F19Sha256 $sourcePath
$compositeSourceHash = $evidenceByProfile['composite'].SourceHash
Assert-F19Evidence (
    $liveSourceHash -eq $compositeSourceHash
) (
    "the live final MEX source is not the captured composite source: " +
    "live=$liveSourceHash composite=$compositeSourceHash."
)
Assert-F19Evidence (
    Test-F19ByteEquality (
        $liveSourceEvidence.Bytes
    ) $evidenceByProfile['composite'].SourceBytes
) 'the live final MEX source is not byte-identical to captured composite.'

if (-not [System.IO.Directory]::Exists($RuntimeRoot)) {
    [void][System.IO.Directory]::CreateDirectory($RuntimeRoot)
}
$sessionName = (
    'schedule-' +
    [System.DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') +
    '-' +
    [System.Guid]::NewGuid().ToString('N').Substring(0, 8)
)
$sessionDirectory = Join-Path $RuntimeRoot $sessionName
[void][System.IO.Directory]::CreateDirectory($sessionDirectory)

$totalProfileVersionRuns = 0
$totalExecutions = 0
$totalVectorAssertions = 0
$totalScalarAssertions = 0
$totalRouteMarkers = 0
$totalReleaseIdentityChecks = 0
$totalMexHashChecks = 0
$profileRunRecords = @()
$runtimeEvidenceByProfile = [ordered]@{}

foreach ($specification in $specifications) {
    $profile = $specification.Name
    $evidence = $evidenceByProfile[$profile]
    $profileRuntime = Join-Path $sessionDirectory $profile
    $summaryPath = Join-Path $profileRuntime 'profile-summary.json'
    $runnerOutputRecords = @(
        & $profileRunner `
            -MexBin $evidence.Mex `
            -SourceSnapshot $evidence.Source `
            -ExpectedMexSha256 $evidence.MexHash `
            -ExpectedSourceSha256 $evidence.SourceHash `
            -Profile $profile `
            -Route $specification.Route `
            -Release both `
            -MatlabR2024b $MatlabR2024b `
            -MatlabR2025b $MatlabR2025b `
            -RuntimeDirectory $profileRuntime `
            -SummaryPath $summaryPath 6>&1
    )
    $runnerOutputLines = @(
        foreach ($record in $runnerOutputRecords) {
            $record.ToString()
        }
    )
    $runnerOutputText = [string]::Join("`n", $runnerOutputLines)
    if ($runnerOutputText.Length -gt 0) {
        $runnerOutputText += "`n"
        Write-Host -NoNewline $runnerOutputText
    }
    Assert-F19Evidence (
        $runnerOutputText.IndexOf(
            'PARTIAL',
            [System.StringComparison]::OrdinalIgnoreCase
        ) -lt 0
    ) "profile '$profile' runner emitted PARTIAL output."

    $routesPerVersion = if ($specification.Route -eq 'all') { 4 } else { 1 }
    $expectedVectorAssertions = $specification.Executions * 8
    $expectedScalarAssertions = $specification.Executions * 5
    $finalMarker = (
        "F19_MATLAB_PROFILE_FINAL profile=$profile " +
        "requested_route=$($specification.Route) versions=2/2 " +
        'release_identity_checks=2/2 ' +
        'observed_releases=R2024b,R2025b ' +
        "routes_per_version=$routesPerVersion/$routesPerVersion " +
        "executions=$($specification.Executions)/" +
        "$($specification.Executions) " +
        "vector_assertions=$expectedVectorAssertions/" +
        "$expectedVectorAssertions " +
        'minimum_vector_assertions_per_execution=8 ' +
        "scalar_assertions=$expectedScalarAssertions/" +
        "$expectedScalarAssertions " +
        'mex_hash_checks=4/4 ' +
        "mex_sha256=$($evidence.MexHash) " +
        "source_sha256=$($evidence.SourceHash) skips=0 " +
        "profile_verdict=FINAL summary=$summaryPath"
    )
    $finalMarkerCount = (
        [System.Text.RegularExpressions.Regex]::Matches(
            $runnerOutputText,
            (
                '(?m)^' +
                [System.Text.RegularExpressions.Regex]::Escape($finalMarker) +
                '\r?$'
            )
        ).Count
    )
    Assert-F19Evidence (
        $finalMarkerCount -eq 1
    ) (
        "profile '$profile' runner emitted $finalMarkerCount exact FINAL " +
        'markers; expected one.'
    )

    Assert-F19Evidence (
        [System.IO.File]::Exists($summaryPath)
    ) "profile '$profile' runner did not write its summary."
    $summary = (
        [System.IO.File]::ReadAllText($summaryPath) |
            ConvertFrom-Json
    )
    Assert-F19Evidence (
        $summary.schema -eq 'dtwc.f19.matlab-profile-run.v1'
    ) "profile '$profile' run-summary schema drift."
    Assert-F19Evidence (
        $summary.profile -eq $profile
    ) "profile '$profile' run-summary identity drift."
    Assert-F19Evidence (
        $summary.requested_route -eq $specification.Route
    ) "profile '$profile' run-summary route drift."
    $requestedReleaseNames = @($summary.requested_releases)
    $observedReleaseNames = @($summary.observed_releases)
    Assert-F19Evidence (
        $requestedReleaseNames.Count -eq 2 -and
        $requestedReleaseNames[0] -eq 'R2024b' -and
        $requestedReleaseNames[1] -eq 'R2025b'
    ) "profile '$profile' requested-release ledger drift."
    Assert-F19Evidence (
        $observedReleaseNames.Count -eq 2 -and
        $observedReleaseNames[0] -eq 'R2024b' -and
        $observedReleaseNames[1] -eq 'R2025b'
    ) "profile '$profile' did not observe both fixed MATLAB releases."
    Assert-F19Evidence (
        [int]$summary.versions -eq 2
    ) "profile '$profile' version count drift."
    Assert-F19Evidence (
        [int]$summary.release_identity_checks -eq 2
    ) "profile '$profile' release-identity check count drift."
    Assert-F19Evidence (
        [int]$summary.mex_hash_checks -eq 4
    ) "profile '$profile' MEX hash-check count drift."
    Assert-F19Evidence (
        [int]$summary.executions -eq $specification.Executions
    ) "profile '$profile' execution count drift."
    Assert-F19Evidence (
        [int]$summary.vector_assertions -eq
            ($specification.Executions * 8)
    ) "profile '$profile' vector-assertion count drift."
    Assert-F19Evidence (
        [int]$summary.minimum_vector_assertions_per_execution -eq 8
    ) "profile '$profile' vector-assertion floor drift."
    Assert-F19Evidence (
        [int]$summary.scalar_assertions -eq
            ($specification.Executions * 5)
    ) "profile '$profile' scalar-assertion count drift."
    Assert-F19Evidence (
        [int]$summary.scalar_assertions_per_execution -eq 5
    ) "profile '$profile' scalar-assertion floor drift."
    Assert-F19Evidence (
        [int]$summary.route_markers -eq $specification.Executions
    ) "profile '$profile' route-marker count drift."
    Assert-F19Evidence (
        [int]$summary.skips -eq 0
    ) "profile '$profile' reported a skip."
    Assert-F19Evidence (
        $summary.profile_verdict -eq 'FINAL'
    ) "profile '$profile' run-summary verdict is not FINAL."
    Assert-F19Evidence (
        ([string]$summary.source_sha256).ToUpperInvariant() -eq
            $evidence.SourceHash
    ) "profile '$profile' run used a different source snapshot."
    Assert-F19Evidence (
        ([string]$summary.mex_sha256).ToUpperInvariant() -eq
            $evidence.MexHash
    ) "profile '$profile' run used a different MEX."
    Assert-F19Evidence (
        [string]::Equals(
            [System.IO.Path]::GetFullPath([string]$summary.source_snapshot),
            $evidence.Source,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) "profile '$profile' run-summary source path drift."
    Assert-F19Evidence (
        [string]::Equals(
            [System.IO.Path]::GetFullPath([string]$summary.mex_path),
            $evidence.Mex,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) "profile '$profile' run-summary MEX path drift."
    Assert-F19Evidence (
        @($summary.release_records).Count -eq 2
    ) "profile '$profile' release-record count drift."
    $releaseRecords = @($summary.release_records)
    $expectedReleaseRecords = @(
        [pscustomobject]@{
            Name = 'R2024b'
            Executable = $MatlabR2024b
        },
        [pscustomobject]@{
            Name = 'R2025b'
            Executable = $MatlabR2025b
        }
    )
    foreach ($releaseIndex in 0..1) {
        $releaseRecord = $releaseRecords[$releaseIndex]
        $expectedRelease = $expectedReleaseRecords[$releaseIndex]
        Assert-F19Evidence (
            $releaseRecord.requested_release -eq $expectedRelease.Name -and
            $releaseRecord.observed_release -eq $expectedRelease.Name
        ) (
            "profile '$profile' release record $releaseIndex identity drift."
        )
        Assert-F19Evidence (
            [string]::Equals(
                [System.IO.Path]::GetFullPath(
                    [string]$releaseRecord.matlab_executable
                ),
                $expectedRelease.Executable,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        ) (
            "profile '$profile' release record $releaseIndex executable drift."
        )
        Assert-F19Evidence (
            (
                [string]$releaseRecord.mex_sha256_before
            ).ToUpperInvariant() -eq $evidence.MexHash -and
            (
                [string]$releaseRecord.mex_sha256_after
            ).ToUpperInvariant() -eq $evidence.MexHash
        ) (
            "profile '$profile' release record $releaseIndex MEX hash drift."
        )
        $integrityMarker = (
            "F19_MATLAB_INTEGRITY requested_release=" +
            "$($expectedRelease.Name) observed_release=" +
            "$($expectedRelease.Name) " +
            "mex_sha256_before=$($evidence.MexHash) " +
            "mex_sha256_after=$($evidence.MexHash) verdict=PASS"
        )
        Assert-F19Evidence (
            (
                [System.Text.RegularExpressions.Regex]::Matches(
                    $runnerOutputText,
                    (
                        '(?m)^' +
                        [System.Text.RegularExpressions.Regex]::Escape(
                            $integrityMarker
                        ) +
                        '\r?$'
                    )
                ).Count
            ) -eq 1
        ) (
            "profile '$profile' release '$($expectedRelease.Name)' exact " +
            'integrity marker is absent or duplicated.'
        )
    }
    $runtimeEvidenceByProfile[$profile] = [pscustomobject]@{
        SummaryPath = $summaryPath
        SummaryHash = Get-F19Sha256 $summaryPath
        ReleaseLogs = @(
            foreach ($releaseRecord in $releaseRecords) {
                [pscustomobject]@{
                    Path = [System.IO.Path]::GetFullPath(
                        [string]$releaseRecord.log
                    )
                    Hash = (
                        [string]$releaseRecord.log_sha256
                    ).ToUpperInvariant()
                }
            }
        )
    }

    $totalProfileVersionRuns += [int]$summary.versions
    $totalExecutions += [int]$summary.executions
    $totalVectorAssertions += [int]$summary.vector_assertions
    $totalScalarAssertions += [int]$summary.scalar_assertions
    $totalRouteMarkers += [int]$summary.route_markers
    $totalReleaseIdentityChecks += [int]$summary.release_identity_checks
    $totalMexHashChecks += [int]$summary.mex_hash_checks
    $profileRunRecords += [ordered]@{
        profile = $profile
        route = $specification.Route
        summary = $summaryPath
        summary_sha256 = Get-F19Sha256 $summaryPath
        source_sha256 = $evidence.SourceHash
        mex_sha256 = $evidence.MexHash
        observed_releases = $observedReleaseNames
        release_identity_checks = [int]$summary.release_identity_checks
        mex_hash_checks = [int]$summary.mex_hash_checks
        executions = [int]$summary.executions
        vector_assertions = [int]$summary.vector_assertions
        scalar_assertions = [int]$summary.scalar_assertions
        skips = [int]$summary.skips
    }
}

$postRunEvidenceRehashes = 0
$postRunEvidenceRecords = @()
foreach ($specification in $specifications) {
    $profile = $specification.Name
    $evidence = $evidenceByProfile[$profile]
    $postSourceEvidence = Read-F19StrictUtf8NoBom (
        $evidence.Source
    ) "post-run profile '$profile' source"
    $postSourceHash = Get-F19Sha256 $evidence.Source
    $postMexHash = Get-F19Sha256 $evidence.Mex
    $postBuildLogHash = Get-F19Sha256 $evidence.BuildLog
    $postManifestHash = Get-F19Sha256 $evidence.ManifestPath
    $runtimeEvidence = $runtimeEvidenceByProfile[$profile]
    $postSummaryHash = Get-F19Sha256 $runtimeEvidence.SummaryPath
    Assert-F19Evidence (
        $postSourceHash -eq $evidence.SourceHash -and
        (
            Test-F19ByteEquality (
                $postSourceEvidence.Bytes
            ) $evidence.SourceBytes
        )
    ) "profile '$profile' source changed during MATLAB schedule."
    Assert-F19Evidence (
        $postMexHash -eq $evidence.MexHash
    ) "profile '$profile' MEX changed during MATLAB schedule."
    Assert-F19Evidence (
        $postBuildLogHash -eq $evidence.BuildLogHash
    ) "profile '$profile' build log changed during MATLAB schedule."
    Assert-F19Evidence (
        $postManifestHash -eq $evidence.ManifestHash
    ) "profile '$profile' manifest changed during MATLAB schedule."
    Assert-F19Evidence (
        $postSummaryHash -eq $runtimeEvidence.SummaryHash
    ) "profile '$profile' run summary changed during MATLAB schedule."
    $postReleaseLogHashes = @()
    foreach ($releaseLog in $runtimeEvidence.ReleaseLogs) {
        $postReleaseLogHash = Get-F19Sha256 $releaseLog.Path
        Assert-F19Evidence (
            $postReleaseLogHash -eq $releaseLog.Hash
        ) "profile '$profile' MATLAB log changed during schedule."
        $postReleaseLogHashes += $postReleaseLogHash
    }
    Assert-F19Evidence (
        $postReleaseLogHashes.Count -eq 2
    ) "profile '$profile' post-run MATLAB-log count drift."
    $postRunEvidenceRehashes += 7
    $postRunEvidenceRecords += [ordered]@{
        profile = $profile
        source_sha256 = $postSourceHash
        mex_sha256 = $postMexHash
        build_log_sha256 = $postBuildLogHash
        manifest_sha256 = $postManifestHash
        summary_sha256 = $postSummaryHash
        matlab_log_sha256 = $postReleaseLogHashes
        rehashes = 7
    }
}
$liveSourceEvidenceAfter = Read-F19StrictUtf8NoBom (
    $sourcePath
) 'post-run live composite MEX source'
$liveSourceHashAfter = Get-F19Sha256 $sourcePath
Assert-F19Evidence (
    $liveSourceHashAfter -eq $compositeSourceHash -and
    (
        Test-F19ByteEquality (
            $liveSourceEvidenceAfter.Bytes
        ) $evidenceByProfile['composite'].SourceBytes
    )
) 'live composite source changed during MATLAB schedule.'

Assert-F19Evidence (
    $specifications.Count -eq 6
) 'fixed profile count is not six.'
Assert-F19Evidence (
    $totalProfileVersionRuns -eq 12
) (
    "profile-version run count drift: observed " +
    "$totalProfileVersionRuns, expected 12."
)
Assert-F19Evidence (
    $totalReleaseIdentityChecks -eq 12
) (
    "release-identity check count drift: observed " +
    "$totalReleaseIdentityChecks, expected 12."
)
Assert-F19Evidence (
    $totalMexHashChecks -eq 24
) (
    "per-invocation MEX hash-check count drift: observed " +
    "$totalMexHashChecks, expected 24."
)
Assert-F19Evidence (
    $totalExecutions -eq 24
) "execution count drift: observed $totalExecutions, expected 24."
Assert-F19Evidence (
    $totalRouteMarkers -eq 24
) "route-marker count drift: observed $totalRouteMarkers, expected 24."
Assert-F19Evidence (
    $totalVectorAssertions -eq 192
) (
    "vector-assertion count drift: observed $totalVectorAssertions, " +
    'expected 192.'
)
Assert-F19Evidence (
    $totalScalarAssertions -eq 120
) (
    "scalar-assertion count drift: observed $totalScalarAssertions, " +
    'expected 120.'
)
Assert-F19Evidence (
    $compileLogCount -eq 6
) "compile-log count drift: observed $compileLogCount, expected 6."
Assert-F19Evidence (
    $semanticProfileCount -eq 6
) (
    "semantic-profile count drift: observed $semanticProfileCount, " +
    'expected 6.'
)
Assert-F19Evidence (
    $postRunEvidenceRehashes -eq 42
) (
    "post-run evidence rehash count drift: observed " +
    "$postRunEvidenceRehashes, expected 42."
)

$aggregate = [ordered]@{
    schema = 'dtwc.f19.matlab-schedule.v1'
    session = $sessionName
    profiles = 6
    requested_releases = @('R2024b', 'R2025b')
    observed_releases = @('R2024b', 'R2025b')
    matlab_executables = @($MatlabR2024b, $MatlabR2025b)
    profile_version_runs = $totalProfileVersionRuns
    release_identity_checks = $totalReleaseIdentityChecks
    mex_hash_checks = $totalMexHashChecks
    executions = $totalExecutions
    route_markers = $totalRouteMarkers
    vector_assertions = $totalVectorAssertions
    minimum_vector_assertions_per_execution = 8
    scalar_assertions = $totalScalarAssertions
    scalar_assertions_per_execution = 5
    unique_source_sha256 = $uniqueSourceHashes.Count
    unique_mex_sha256 = $uniqueMexHashes.Count
    compile_logs = $compileLogCount
    semantic_profiles = $semanticProfileCount
    live_source_sha256_before = $liveSourceHash
    live_source_sha256_after = $liveSourceHashAfter
    composite_source_sha256 = $compositeSourceHash
    post_run_evidence_rehashes = $postRunEvidenceRehashes
    skips = 0
    verdict = 'PASS'
    profile_runs = $profileRunRecords
    post_run_evidence = $postRunEvidenceRecords
}
$aggregatePath = Join-Path $sessionDirectory 'schedule-summary.json'
$aggregateText = $aggregate | ConvertTo-Json -Depth 8
Write-F19EvidenceUtf8NoBom (
    $aggregatePath
) ($aggregateText + "`n") $repositoryRoot

Write-Host (
    'F19_MATLAB_SCHEDULE profiles=6/6 versions=2/2 ' +
    "release_identity_checks=$totalReleaseIdentityChecks/12 " +
    'observed_releases=R2024b,R2025b ' +
    "profile_version_runs=$totalProfileVersionRuns/12 " +
    "executions=$totalExecutions/24 " +
    "route_markers=$totalRouteMarkers/24 " +
    "vector_assertions=$totalVectorAssertions/192 " +
    'minimum_vector_assertions_per_execution=8 ' +
    "scalar_assertions=$totalScalarAssertions/120 " +
    "mex_hash_checks=$totalMexHashChecks/24 " +
    "unique_source_sha256=$($uniqueSourceHashes.Count)/6 " +
    "unique_mex_sha256=$($uniqueMexHashes.Count)/6 " +
    "compile_logs=$compileLogCount/6 " +
    "semantic_profiles=$semanticProfileCount/6 " +
    "post_run_evidence_rehashes=$postRunEvidenceRehashes/42 " +
    'live_composite_rehashes=1/1 skips=0 ' +
    "summary=$aggregatePath verdict=PASS"
)
