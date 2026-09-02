<#
.SYNOPSIS
Preflight, execute, or recover the registered F20 storage-policy mutations.

.DESCRIPTION
Preflight is read-only and is the default. Execute temporarily materializes
exactly one mutant at a time, builds only unit_test_problem_storage_policy,
requires the mutant binary to exit nonzero without a PASS/skip marker, and
restores byte-exact source snapshots in finally blocks. Execute is permitted
only with -ConfirmExclusiveBuildAccess after every concurrent native build has
finished. A disk recovery snapshot remains under the canonical build tree for
hard-process-loss recovery through -Mode Restore.

Registered decisive band: 11/11 executable mutants KILLED; initial and final
llfio-ON/OFF controls pass with zero subject skips and at least 963/606
assertions across exactly five cases.

.EXAMPLE
.\tests\mutation\f20_problem_storage_policy_mutations.ps1 -Mode Preflight

.EXAMPLE
.\tests\mutation\f20_problem_storage_policy_mutations.ps1 `
    -Mode Execute -ConfirmExclusiveBuildAccess

.EXAMPLE
.\tests\mutation\f20_problem_storage_policy_mutations.ps1 -Mode Restore
#>

[CmdletBinding()]
param(
    [ValidateSet('Preflight', 'Execute', 'Restore')]
    [string] $Mode = 'Preflight',

    [string] $RepoRoot = '',

    [string] $CanonicalBuild = 'build/highs-1151',

    [string] $NoLlfioBuild = 'build/nollfio',

    [switch] $ConfirmExclusiveBuildAccess
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ExpectedMutationCount = 11
$TargetName = 'unit_test_problem_storage_policy'
$PassMarkerPrefix = 'F20_PROBLEM_STORAGE_POLICY'
$SkipDiagnosticPattern = '[Ss][Kk][Ii][Pp]([Pp]|[ :])'
$Utf8Strict = New-Object System.Text.UTF8Encoding($false, $true)

function Assert-F20
{
    param(
        [bool] $Condition,
        [string] $Message
    )

    if (-not $Condition) {
        throw "F20 mutation harness: $Message"
    }
}

function Get-NormalizedFullPath
{
    param(
        [string] $Path,
        [string] $BasePath
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($BasePath, $Path))
}

function Assert-PathInsideRepo
{
    param(
        [string] $Path,
        [string] $Root,
        [string] $Label
    )

    $separator = [System.IO.Path]::DirectorySeparatorChar
    $rootPrefix = $Root.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar) + $separator
    Assert-F20 (
        $Path.StartsWith(
            $rootPrefix,
            [System.StringComparison]::OrdinalIgnoreCase)
    ) "$Label escapes the repository root: $Path"
}

function Get-Sha256Bytes
{
    param([byte[]] $Bytes)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString(
            $sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Get-FileSha256
{
    param([string] $Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).
        Hash.ToLowerInvariant()
}

function Invoke-NativeCapture
{
    param(
        [string] $FilePath,
        [string[]] $Arguments
    )

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $captured = @(
            & $FilePath @Arguments 2>&1 |
                ForEach-Object { $_.ToString() }
        )
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Lines = [string[]] $captured
        Text = [string]::Join([Environment]::NewLine, $captured)
    }
}

function Write-NativeOutput
{
    param(
        [string] $Header,
        [pscustomobject] $Result
    )

    Write-Host $Header
    foreach ($line in $Result.Lines) {
        Write-Host $line
    }
}

function Get-ExactOccurrenceCount
{
    param(
        [string] $Text,
        [string] $Needle
    )

    Assert-F20 ($Needle.Length -gt 0) 'empty exact mutation needle'
    $count = 0
    $offset = 0
    while ($offset -le $Text.Length - $Needle.Length) {
        $index = $Text.IndexOf(
            $Needle,
            $offset,
            [System.StringComparison]::Ordinal)
        if ($index -lt 0) {
            break
        }
        ++$count
        $offset = $index + $Needle.Length
    }
    return $count
}

function Replace-ExactOnce
{
    param(
        [string] $Text,
        [string] $Needle,
        [string] $Replacement,
        [string] $MutationId
    )

    $count = Get-ExactOccurrenceCount -Text $Text -Needle $Needle
    Assert-F20 ($count -eq 1) (
        "$MutationId exact precondition matched $count times, expected 1")
    $index = $Text.IndexOf(
        $Needle,
        [System.StringComparison]::Ordinal)
    return $Text.Substring(0, $index) +
        $Replacement +
        $Text.Substring($index + $Needle.Length)
}

function Replace-RegexOnce
{
    param(
        [string] $Text,
        [string] $Pattern,
        [string] $ReplacementTemplate,
        [string] $MutationId
    )

    $regex = New-Object System.Text.RegularExpressions.Regex(
        $Pattern,
        [System.Text.RegularExpressions.RegexOptions]::CultureInvariant)
    $matches = $regex.Matches($Text)
    Assert-F20 ($matches.Count -eq 1) (
        "$MutationId regex precondition matched $($matches.Count) times, expected 1")
    $match = $matches[0]
    $replacement = $match.Result($ReplacementTemplate)
    $eol = "`n"
    if ($match.Value.Contains("`r`n")) {
        $eol = "`r`n"
    }
    $replacement = $replacement.Replace('{EOL}', $eol)
    return $Text.Substring(0, $match.Index) +
        $replacement +
        $Text.Substring($match.Index + $match.Length)
}

function Apply-MutationInMemory
{
    param(
        [string] $Text,
        [pscustomobject] $Mutation
    )

    switch ($Mutation.Kind) {
    'Exact' {
        return Replace-ExactOnce `
            -Text $Text `
            -Needle $Mutation.Needle `
            -Replacement $Mutation.Replacement `
            -MutationId $Mutation.Id
    }
    'Regex' {
        return Replace-RegexOnce `
            -Text $Text `
            -Pattern $Mutation.Pattern `
            -ReplacementTemplate $Mutation.Replacement `
            -MutationId $Mutation.Id
    }
    default {
        throw "F20 mutation harness: unsupported mutation kind '$($Mutation.Kind)'"
    }
    }
}

function Assert-SourceHashes
{
    param(
        [hashtable] $Snapshots,
        [string[]] $SourceFiles,
        [string] $Stage
    )

    foreach ($relativePath in $SourceFiles) {
        $snapshot = $Snapshots[$relativePath]
        $observed = Get-FileSha256 -Path $snapshot.Path
        Assert-F20 ($observed -eq $snapshot.Hash) (
            "$Stage source hash drift for $relativePath; " +
            "expected=$($snapshot.Hash) observed=$observed")
    }
}

function Restore-SourceSnapshots
{
    param(
        [hashtable] $Snapshots,
        [string[]] $SourceFiles
    )

    foreach ($relativePath in $SourceFiles) {
        $snapshot = $Snapshots[$relativePath]
        $observed = Get-FileSha256 -Path $snapshot.Path
        if ($observed -ne $snapshot.Hash) {
            [System.IO.File]::WriteAllBytes($snapshot.Path, $snapshot.Bytes)
        }
    }
    Assert-SourceHashes `
        -Snapshots $Snapshots `
        -SourceFiles $SourceFiles `
        -Stage 'restore'
}

function Write-RecoverySnapshot
{
    param(
        [hashtable] $Snapshots,
        [string[]] $SourceFiles,
        [string] $RecoveryRoot,
        [string] $Root,
        [string] $HeadCommit
    )

    [System.IO.Directory]::CreateDirectory($RecoveryRoot) | Out-Null
    $records = @()
    foreach ($relativePath in $SourceFiles) {
        $snapshot = $Snapshots[$relativePath]
        $snapshotName = $relativePath.Replace('/', '__').Replace('\', '__') +
            '.snapshot'
        $snapshotPath = [System.IO.Path]::Combine(
            $RecoveryRoot,
            $snapshotName)
        [System.IO.File]::WriteAllBytes($snapshotPath, $snapshot.Bytes)
        $snapshotHash = Get-FileSha256 -Path $snapshotPath
        Assert-F20 ($snapshotHash -eq $snapshot.Hash) (
            "recovery snapshot hash mismatch for $relativePath")
        $records += [pscustomobject]@{
            relative_path = $relativePath
            source_sha256 = $snapshot.Hash
            snapshot_file = $snapshotName
        }
    }

    $manifest = [pscustomobject]@{
        schema = 'dtwc-f20-mutation-recovery-v1'
        repo_root = $Root
        head_commit = $HeadCommit
        files = $records
    }
    $manifestText = $manifest | ConvertTo-Json -Depth 5
    $manifestPath = [System.IO.Path]::Combine(
        $RecoveryRoot,
        'manifest.json')
    [System.IO.File]::WriteAllText(
        $manifestPath,
        $manifestText,
        $Utf8Strict)
    Write-Output "F20_MUTATION_RECOVERY path=$manifestPath files=$($records.Count)"
}

function Restore-RecoverySnapshot
{
    param(
        [string] $RecoveryRoot,
        [string] $Root
    )

    $manifestPath = [System.IO.Path]::Combine(
        $RecoveryRoot,
        'manifest.json')
    Assert-F20 ([System.IO.File]::Exists($manifestPath)) (
        "recovery manifest does not exist: $manifestPath")
    $manifest = Get-Content -LiteralPath $manifestPath -Raw |
        ConvertFrom-Json
    Assert-F20 ($manifest.schema -eq 'dtwc-f20-mutation-recovery-v1') (
        "unexpected recovery schema '$($manifest.schema)'")
    $manifestRoot = [System.IO.Path]::GetFullPath(
        [string] $manifest.repo_root)
    Assert-F20 ($manifestRoot -eq $Root) (
        "recovery manifest repo root mismatch: $manifestRoot")

    $allowedFiles = @(
        'dtwc/Problem.hpp',
        'dtwc/DataLoader.hpp',
        'dtwc/Problem.cpp'
    )
    $manifestFiles = @(
        $manifest.files |
            ForEach-Object { [string] $_.relative_path }
    )
    Assert-F20 ($manifestFiles.Count -eq $allowedFiles.Count) (
        "recovery manifest file count=$($manifestFiles.Count), " +
        "expected=$($allowedFiles.Count)")
    foreach ($allowedFile in $allowedFiles) {
        Assert-F20 ($manifestFiles -contains $allowedFile) (
            "recovery manifest is missing $allowedFile")
    }

    foreach ($record in $manifest.files) {
        $relativePath = [string] $record.relative_path
        $targetPath = Get-NormalizedFullPath -Path $relativePath -BasePath $Root
        Assert-PathInsideRepo `
            -Path $targetPath `
            -Root $Root `
            -Label "recovery target $relativePath"
        $snapshotPath = Get-NormalizedFullPath `
            -Path ([string] $record.snapshot_file) `
            -BasePath $RecoveryRoot
        Assert-PathInsideRepo `
            -Path $snapshotPath `
            -Root $Root `
            -Label "recovery snapshot $relativePath"
        $bytes = [System.IO.File]::ReadAllBytes($snapshotPath)
        $hash = Get-Sha256Bytes -Bytes $bytes
        Assert-F20 ($hash -eq [string] $record.source_sha256) (
            "recovery snapshot is corrupt for $relativePath")
        [System.IO.File]::WriteAllBytes($targetPath, $bytes)
        $restoredHash = Get-FileSha256 -Path $targetPath
        Assert-F20 ($restoredHash -eq [string] $record.source_sha256) (
            "recovery restore hash mismatch for $relativePath")
        Write-Output (
            "F20_MUTATION_RESTORE file=$relativePath sha256=$restoredHash")
    }
    Write-Output 'F20_MUTATION_RESTORE verdict=PASS'
}

function Invoke-Build
{
    param(
        [pscustomobject] $Profile,
        [string] $Label
    )

    $result = Invoke-NativeCapture `
        -FilePath 'cmake' `
        -Arguments @(
            '--build',
            $Profile.BuildRoot,
            '--target',
            $TargetName)
    Write-NativeOutput `
        -Header "F20_MUTATION_BUILD label=$Label profile=$($Profile.Name)" `
        -Result $result
    return $result
}

function Invoke-FocusedBinary
{
    param(
        [pscustomobject] $Profile,
        [string] $Label
    )

    [System.IO.Directory]::CreateDirectory($Profile.TestRoot) | Out-Null
    $variables = @('TMP', 'TEMP', 'TMPDIR')
    $previous = @{}
    foreach ($variable in $variables) {
        $previous[$variable] = [Environment]::GetEnvironmentVariable(
            $variable,
            [EnvironmentVariableTarget]::Process)
        [Environment]::SetEnvironmentVariable(
            $variable,
            $Profile.TestRoot,
            [EnvironmentVariableTarget]::Process)
    }

    try {
        $result = Invoke-NativeCapture `
            -FilePath $Profile.Binary `
            -Arguments @()
    }
    finally {
        foreach ($variable in $variables) {
            [Environment]::SetEnvironmentVariable(
                $variable,
                $previous[$variable],
                [EnvironmentVariableTarget]::Process)
        }
    }

    Write-NativeOutput `
        -Header "F20_MUTATION_RUN label=$Label profile=$($Profile.Name)" `
        -Result $result
    return $result
}

function Assert-CleanControl
{
    param(
        [pscustomobject] $Profile,
        [string] $Label
    )

    $build = Invoke-Build -Profile $Profile -Label $Label
    Assert-F20 ($build.ExitCode -eq 0) (
        "$Label $($Profile.Name) control build failed with exit $($build.ExitCode)")
    Assert-F20 ([System.IO.File]::Exists($Profile.Binary)) (
        "$Label $($Profile.Name) binary is absent: $($Profile.Binary)")
    $binaryHash = Get-FileSha256 -Path $Profile.Binary
    $run = Invoke-FocusedBinary -Profile $Profile -Label $Label
    Assert-F20 ($run.ExitCode -eq 0) (
        "$Label $($Profile.Name) control exited $($run.ExitCode)")
    Assert-F20 ($run.Text.Contains($Profile.Marker)) (
        "$Label $($Profile.Name) control marker is absent")
    Assert-F20 (-not [regex]::IsMatch(
        $run.Text,
        $SkipDiagnosticPattern)) (
        "$Label $($Profile.Name) control emitted a skip diagnostic")

    $assertionPattern =
        'All tests passed \((\d+) assertions in 5 test cases\)'
    $assertionMatch = [regex]::Match($run.Text, $assertionPattern)
    Assert-F20 ($assertionMatch.Success) (
        "$Label $($Profile.Name) assertion/case summary is absent")
    $assertions = [int] $assertionMatch.Groups[1].Value
    Assert-F20 ($assertions -ge $Profile.AssertionFloor) (
        "$Label $($Profile.Name) assertions=$assertions below " +
        "registered floor=$($Profile.AssertionFloor)")
    Write-Host (
        "F20_MUTATION_CONTROL label=$Label profile=$($Profile.Name) " +
        "assertions=$assertions cases=5 skips=0 sha256=$binaryHash verdict=PASS")
    return $binaryHash
}

function Assert-MutantKilled
{
    param(
        [pscustomobject] $Mutation,
        [pscustomobject] $Profile,
        [string] $ControlBinaryHash
    )

    $build = Invoke-Build -Profile $Profile -Label $Mutation.Id
    Assert-F20 ($build.ExitCode -eq 0) (
        "$($Mutation.Id) was not an executable mutant: build exit " +
        "$($build.ExitCode)")
    Assert-F20 ([System.IO.File]::Exists($Profile.Binary)) (
        "$($Mutation.Id) binary is absent after its build")
    $mutantBinaryHash = Get-FileSha256 -Path $Profile.Binary
    Assert-F20 ($mutantBinaryHash -ne $ControlBinaryHash) (
        "$($Mutation.Id) did not change the focused binary hash")

    $run = Invoke-FocusedBinary -Profile $Profile -Label $Mutation.Id
    Assert-F20 ($run.ExitCode -ne 0) (
        "$($Mutation.Id) SURVIVED with process exit 0")
    Assert-F20 (-not $run.Text.Contains($PassMarkerPrefix)) (
        "$($Mutation.Id) emitted an F20 passing marker despite failing")
    Assert-F20 (-not [regex]::IsMatch(
        $run.Text,
        $SkipDiagnosticPattern)) (
        "$($Mutation.Id) was classified by a skip diagnostic")
    Write-Host (
        "F20_MUTATION id=$($Mutation.Id) profile=$($Profile.Name) " +
        "build=passed test_exit=$($run.ExitCode) marker=absent skips=0 " +
        "binary_sha256=$mutantBinaryHash verdict=KILLED")
}

$Mutations = @(
    [pscustomobject]@{
        Id = 'M01'
        Profile = 'llfio-on'
        RelativePath = 'dtwc/Problem.hpp'
        Kind = 'Regex'
        Pattern = '(?m)^    auto loaded = detail::route_series_storage\(\r?\n      std::move\(candidate\),\r?\n      storage_policy_,\r?\n      ram_limit_bytes_,\r?\n      \{\},\r?\n      "Problem::set_data"\);\r?\n    adopt_loaded_data\(std::move\(loaded\)\);'
        Replacement = '    data_ = std::move(candidate);{EOL}    series_storage_owner_.reset();'
        Description = 'restore advisory-only Problem setter behavior'
    }
    [pscustomobject]@{
        Id = 'M02'
        Profile = 'llfio-on'
        RelativePath = 'dtwc/DataLoader.hpp'
        Kind = 'Regex'
        Pattern = '\|\| \(policy == core::StoragePolicy::Auto\r?\n        && choose_storage\(footprint, available, ram_limit_bytes\)\r?\n             == core::StoragePolicy::Mmap\);'
        Replacement = '|| choose_storage(footprint, available, ram_limit_bytes) == core::StoragePolicy::Mmap;'
        Description = 'ignore Heap in the shared routing predicate'
    }
    [pscustomobject]@{
        Id = 'M03'
        Profile = 'llfio-on'
        RelativePath = 'dtwc/DataLoader.hpp'
        Kind = 'Regex'
        Pattern = '(?m)(^  const bool want_mmap =\r?\n    )policy == core::StoragePolicy::Mmap(\r?\n    \|\| \(policy == core::StoragePolicy::Auto\r?\n        && choose_storage\(footprint, available, ram_limit_bytes\)\r?\n             == core::StoragePolicy::Mmap\);)'
        Replacement = '$1false$2'
        Description = 'ignore explicit Mmap in the shared routing predicate'
    }
    [pscustomobject]@{
        Id = 'M04'
        Profile = 'llfio-on'
        RelativePath = 'dtwc/Problem.hpp'
        Kind = 'Exact'
        Needle = 'series_storage_owner_ = std::move(owner);'
        Replacement = 'series_storage_owner_.reset();'
        Description = 'publish mapped Data without retaining LoadedData'
    }
    [pscustomobject]@{
        Id = 'M05'
        Profile = 'llfio-on'
        RelativePath = 'dtwc/Problem.hpp'
        Kind = 'Exact'
        Needle = 'data_ = std::move(view);'
        Replacement = 'owner->names.clear(); data_ = std::move(view);'
        Description = 'drop mapped name ownership after the adoption preflight'
    }
    [pscustomobject]@{
        Id = 'M06'
        Profile = 'llfio-on'
        RelativePath = 'dtwc/Problem.hpp'
        Kind = 'Exact'
        Needle = 'Problem(Problem &&) = default;'
        Replacement = 'Problem(Problem &&other) : Problem() { *this = std::move(other); other.series_storage_owner_ = std::move(series_storage_owner_); }'
        Description = 'leave mapped ownership in the move source'
    }
    [pscustomobject]@{
        Id = 'M07'
        Profile = 'llfio-on'
        RelativePath = 'dtwc/Problem.hpp'
        Kind = 'Regex'
        Pattern = '(?m)^    data_ = std::move\(candidate\);\r?\n    series_storage_owner_\.reset\(\);\r?\n    refresh_distance_matrix\(\);\r?\n    resize\(\);[^\r\n]*'
        Replacement = '    set_data(std::move(candidate));{EOL}    resize();'
        Description = 'route set_view_data through owning policy storage'
    }
    [pscustomobject]@{
        Id = 'M08'
        Profile = 'llfio-off'
        RelativePath = 'dtwc/DataLoader.hpp'
        Kind = 'Regex'
        Pattern = '(?m)(^#else\r?\n  )if \(policy == core::StoragePolicy::Mmap\)( \{\r?\n    throw IOError\()'
        Replacement = '$1if (false)$2'
        Description = 'silently Heap-fallback explicit Mmap without llfio'
    }
    [pscustomobject]@{
        Id = 'M09'
        Profile = 'llfio-on'
        RelativePath = 'dtwc/DataLoader.hpp'
        Kind = 'Regex'
        Pattern = '(?m)(^  if \(resident\.is_f32\(\)\) \{\r?\n    )if \(policy == core::StoragePolicy::Mmap\)( \{)'
        Replacement = '$1if (false)$2'
        Description = 'silently Heap-fallback explicit Float32 Mmap'
    }
    [pscustomobject]@{
        Id = 'M10'
        Profile = 'llfio-on'
        RelativePath = 'dtwc/Problem.hpp'
        Kind = 'Exact'
        Needle = 'adopt_loaded_data(loader.load_stored());'
        Replacement = 'data_ = loader.load();'
        Description = 'leave Problem loader construction on heap-only load'
    }
    [pscustomobject]@{
        Id = 'M11'
        Profile = 'llfio-on'
        RelativePath = 'dtwc/Problem.cpp'
        Kind = 'Exact'
        Needle = 'if (has_mmap_series_storage()'
        Replacement = 'if (false && has_mmap_series_storage()'
        Description = 'allow mapped empty owning vectors to reach GPU dispatch'
    }
)

Assert-F20 ($Mutations.Count -eq $ExpectedMutationCount) (
    "mutation inventory=$($Mutations.Count), expected=$ExpectedMutationCount")
$mutationIds = @($Mutations | ForEach-Object { $_.Id })
Assert-F20 (($mutationIds | Select-Object -Unique).Count -eq
    $ExpectedMutationCount) 'mutation IDs are not unique'

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Get-NormalizedFullPath `
        -Path ([System.IO.Path]::Combine($PSScriptRoot, '..', '..')) `
        -BasePath $PSScriptRoot
}
else {
    $RepoRoot = Get-NormalizedFullPath `
        -Path $RepoRoot `
        -BasePath (Get-Location).Path
}
Assert-F20 ([System.IO.Directory]::Exists($RepoRoot)) (
    "repository root does not exist: $RepoRoot")
Assert-F20 ($null -ne (Get-Command git -ErrorAction SilentlyContinue)) (
    'git is not available on PATH')

$gitRootResult = Invoke-NativeCapture `
    -FilePath 'git' `
    -Arguments @('-C', $RepoRoot, 'rev-parse', '--show-toplevel')
Assert-F20 ($gitRootResult.ExitCode -eq 0) 'git root probe failed'
$GitRoot = [System.IO.Path]::GetFullPath($gitRootResult.Text.Trim())
Assert-F20 ($GitRoot -eq $RepoRoot) (
    "resolved root '$RepoRoot' is not git root '$GitRoot'")

$CanonicalBuild = Get-NormalizedFullPath `
    -Path $CanonicalBuild `
    -BasePath $RepoRoot
$NoLlfioBuild = Get-NormalizedFullPath `
    -Path $NoLlfioBuild `
    -BasePath $RepoRoot
Assert-PathInsideRepo `
    -Path $CanonicalBuild `
    -Root $RepoRoot `
    -Label 'canonical build'
Assert-PathInsideRepo `
    -Path $NoLlfioBuild `
    -Root $RepoRoot `
    -Label 'llfio-off build'

$RecoveryRoot = [System.IO.Path]::Combine(
    $CanonicalBuild,
    'tests',
    'f20-mutation-recovery')
Assert-PathInsideRepo `
    -Path $RecoveryRoot `
    -Root $RepoRoot `
    -Label 'recovery root'

if ($Mode -eq 'Restore') {
    Restore-RecoverySnapshot `
        -RecoveryRoot $RecoveryRoot `
        -Root $RepoRoot
    return
}

Assert-F20 ($null -ne (Get-Command cmake -ErrorAction SilentlyContinue)) (
    'cmake is not available on PATH')

$exeSuffix = ''
if ([Environment]::OSVersion.Platform -eq
    [PlatformID]::Win32NT) {
    $exeSuffix = '.exe'
}

$Profiles = @{
    'llfio-on' = [pscustomobject]@{
        Name = 'llfio-on'
        BuildRoot = $CanonicalBuild
        TestRoot = [System.IO.Path]::Combine(
            $CanonicalBuild,
            'tests',
            'f20-problem-storage')
        Binary = [System.IO.Path]::Combine(
            $CanonicalBuild,
            'bin',
            $TargetName + $exeSuffix)
        Marker = 'F20_PROBLEM_STORAGE_POLICY build=llfio-on footprint=288 heap=owning mmap=view values=72/72 names=12/12 ndim_routes=2/2 ordered_pairs=72/72 artifact=pass lifetime=pass loader_auto=mmap view_override=pass subject_skips=0 verdict=PASS'
        AssertionFloor = 963
    }
    'llfio-off' = [pscustomobject]@{
        Name = 'llfio-off'
        BuildRoot = $NoLlfioBuild
        TestRoot = [System.IO.Path]::Combine(
            $NoLlfioBuild,
            'tests',
            'f20-problem-storage')
        Binary = [System.IO.Path]::Combine(
            $NoLlfioBuild,
            'bin',
            $TargetName + $exeSuffix)
        Marker = 'F20_PROBLEM_STORAGE_POLICY build=llfio-off footprint=288 heap=owning mmap=rejected values=36/36 names=6/6 ndim_routes=1/1 ordered_pairs=36/36 transaction=pass loader_auto=heap-warning view_override=pass subject_skips=0 verdict=PASS'
        AssertionFloor = 606
    }
}

foreach ($profileName in @('llfio-on', 'llfio-off')) {
    $profile = $Profiles[$profileName]
    Assert-PathInsideRepo `
        -Path $profile.TestRoot `
        -Root $RepoRoot `
        -Label "$profileName test root"
    Assert-PathInsideRepo `
        -Path $profile.Binary `
        -Root $RepoRoot `
        -Label "$profileName test binary"
    Assert-F20 ([System.IO.Directory]::Exists($profile.BuildRoot)) (
        "$profileName build directory is absent: $($profile.BuildRoot)")
}

$SourceFiles = @(
    'dtwc/Problem.hpp',
    'dtwc/DataLoader.hpp',
    'dtwc/Problem.cpp'
)
$Snapshots = @{}
foreach ($relativePath in $SourceFiles) {
    $sourcePath = Get-NormalizedFullPath `
        -Path $relativePath `
        -BasePath $RepoRoot
    Assert-PathInsideRepo `
        -Path $sourcePath `
        -Root $RepoRoot `
        -Label "source $relativePath"
    Assert-F20 ([System.IO.File]::Exists($sourcePath)) (
        "source file is absent: $sourcePath")
    $bytes = [System.IO.File]::ReadAllBytes($sourcePath)
    $hasBom = $bytes.Length -ge 3 -and
        $bytes[0] -eq 0xEF -and
        $bytes[1] -eq 0xBB -and
        $bytes[2] -eq 0xBF
    Assert-F20 (-not $hasBom) (
        "$relativePath unexpectedly has a UTF-8 BOM")
    $text = $Utf8Strict.GetString($bytes)
    $Snapshots[$relativePath] = [pscustomobject]@{
        Path = $sourcePath
        Bytes = $bytes
        Text = $text
        Hash = Get-Sha256Bytes -Bytes $bytes
    }
}

$statusArguments = @(
    '-C',
    $RepoRoot,
    'status',
    '--porcelain',
    '--'
)
$statusArguments += $SourceFiles
$statusResult = Invoke-NativeCapture `
    -FilePath 'git' `
    -Arguments $statusArguments
Assert-F20 ($statusResult.ExitCode -eq 0) 'target-source status probe failed'

Assert-F20 ([string]::IsNullOrWhiteSpace($statusResult.Text)) (
    "refusing to mutate dirty target source files:$([Environment]::NewLine)" +
    $statusResult.Text)

# This exact M01 precondition is also the cheap permanent guard that
# Problem::set_data passes storage_policy_ unchanged to the shared router.
foreach ($mutation in $Mutations) {
    $snapshot = $Snapshots[$mutation.RelativePath]
    $mutatedText = Apply-MutationInMemory `
        -Text $snapshot.Text `
        -Mutation $mutation
    Assert-F20 ($mutatedText -ne $snapshot.Text) (
        "$($mutation.Id) in-memory mutation was a no-op")
    $mutatedBytes = $Utf8Strict.GetBytes($mutatedText)
    $mutatedHash = Get-Sha256Bytes -Bytes $mutatedBytes
    Assert-F20 ($mutatedHash -ne $snapshot.Hash) (
        "$($mutation.Id) in-memory hash did not change")
    Write-Output (
        "F20_MUTATION_PREFLIGHT id=$($mutation.Id) " +
        "profile=$($mutation.Profile) file=$($mutation.RelativePath) " +
        "source_sha256=$($snapshot.Hash) mutant_sha256=$mutatedHash " +
        "precondition=exactly-one verdict=PASS")
}

Write-Output (
    "F20_MUTATION_PREFLIGHT mutations=$ExpectedMutationCount " +
    "problem_policy_passthrough=pass target_sources=clean verdict=PASS")

if ($Mode -eq 'Preflight') {
    return
}

Assert-F20 $ConfirmExclusiveBuildAccess.IsPresent (
    'Execute mode requires -ConfirmExclusiveBuildAccess after all concurrent builds finish')

$headResult = Invoke-NativeCapture `
    -FilePath 'git' `
    -Arguments @('-C', $RepoRoot, 'rev-parse', 'HEAD')
Assert-F20 ($headResult.ExitCode -eq 0) 'HEAD probe failed'
$HeadCommit = $headResult.Text.Trim()
Write-RecoverySnapshot `
    -Snapshots $Snapshots `
    -SourceFiles $SourceFiles `
    -RecoveryRoot $RecoveryRoot `
    -Root $RepoRoot `
    -HeadCommit $HeadCommit

$controlHashes = @{}
$killed = 0
$executionFailure = $null

Push-Location $RepoRoot
try {
    Assert-SourceHashes `
        -Snapshots $Snapshots `
        -SourceFiles $SourceFiles `
        -Stage 'initial control'
    foreach ($profileName in @('llfio-on', 'llfio-off')) {
        $controlHashes[$profileName] = Assert-CleanControl `
            -Profile $Profiles[$profileName] `
            -Label 'initial'
    }

    foreach ($mutation in $Mutations) {
        Assert-SourceHashes `
            -Snapshots $Snapshots `
            -SourceFiles $SourceFiles `
            -Stage "$($mutation.Id) pre-mutation"
        $snapshot = $Snapshots[$mutation.RelativePath]
        $mutatedText = Apply-MutationInMemory `
            -Text $snapshot.Text `
            -Mutation $mutation
        $mutatedBytes = $Utf8Strict.GetBytes($mutatedText)
        [System.IO.File]::WriteAllBytes($snapshot.Path, $mutatedBytes)
        $materializedHash = Get-FileSha256 -Path $snapshot.Path
        Assert-F20 ($materializedHash -ne $snapshot.Hash) (
            "$($mutation.Id) was not materialized")
        Write-Output (
            "F20_MUTATION_MATERIALIZED id=$($mutation.Id) " +
            "file=$($mutation.RelativePath) sha256=$materializedHash")

        try {
            $profile = $Profiles[$mutation.Profile]
            Assert-MutantKilled `
                -Mutation $mutation `
                -Profile $profile `
                -ControlBinaryHash $controlHashes[$mutation.Profile]
            ++$killed
        }
        finally {
            Restore-SourceSnapshots `
                -Snapshots $Snapshots `
                -SourceFiles $SourceFiles
            Write-Output (
                "F20_MUTATION_SOURCE_RESTORE id=$($mutation.Id) verdict=PASS")
        }
    }

    Assert-F20 ($killed -eq $ExpectedMutationCount) (
        "killed=$killed expected=$ExpectedMutationCount")

    foreach ($profileName in @('llfio-on', 'llfio-off')) {
        [void] (Assert-CleanControl `
            -Profile $Profiles[$profileName] `
            -Label 'final')
    }
}
catch {
    $executionFailure = $_
}
finally {
    Restore-SourceSnapshots `
        -Snapshots $Snapshots `
        -SourceFiles $SourceFiles
    if ($null -ne $executionFailure) {
        foreach ($profileName in @('llfio-on', 'llfio-off')) {
            try {
                [void] (Invoke-Build `
                    -Profile $Profiles[$profileName] `
                    -Label 'failure-cleanup')
            }
            catch {
                Write-Warning (
                    "failure cleanup build also failed for ${profileName}: " +
                    $_.Exception.Message)
            }
        }
    }
    Pop-Location
}

if ($null -ne $executionFailure) {
    throw $executionFailure
}

Write-Output (
    "F20_MUTATION_SUMMARY controls=4/4 mutations=$ExpectedMutationCount " +
    "killed=$killed survived=0 source_restore=pass verdict=PASS")
