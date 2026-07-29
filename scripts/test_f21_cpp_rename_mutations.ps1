[CmdletBinding()]
param(
    [string]$BuildDirectory = "build/highs-1151"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-F21 {
    param(
        [bool]$Condition,
        [string]$Message
    )
    if (-not $Condition) {
        throw "F21 mutation gate: $Message"
    }
}

function Get-F21FullPath {
    param(
        [string]$Path,
        [string]$Base
    )
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($Base, $Path))
}

function Assert-F21InsideRepository {
    param(
        [string]$Path,
        [string]$RepositoryRoot,
        [string]$Description
    )
    $trimChars = [char[]]@(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar)
    $rootPrefix = $RepositoryRoot.TrimEnd($trimChars) +
        [System.IO.Path]::DirectorySeparatorChar
    Assert-F21 (
        $Path.StartsWith(
            $rootPrefix,
            [System.StringComparison]::OrdinalIgnoreCase)
    ) "$Description escapes the repository root: $Path"
}

function Get-F21Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function ConvertFrom-F21Utf8 {
    param([byte[]]$Bytes)
    $utf8 = [System.Text.UTF8Encoding]::new($false, $true)
    return $utf8.GetString($Bytes)
}

function ConvertTo-F21Utf8 {
    param([string]$Text)
    $utf8 = [System.Text.UTF8Encoding]::new($false, $true)
    return $utf8.GetBytes($Text)
}

function Replace-F21Exact {
    param(
        [string]$Text,
        [string]$Needle,
        [string]$Replacement,
        [int]$ExpectedCount,
        [string]$MutationName
    )
    $count = [regex]::Matches(
        $Text, [regex]::Escape($Needle)).Count
    Assert-F21 ($count -eq $ExpectedCount) (
        "$MutationName expected $ExpectedCount occurrence(s) of " +
        "'$Needle', observed $count"
    )
    return $Text.Replace($Needle, $Replacement)
}

function Invoke-F21Native {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $raw = & $Executable @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    $text = ($raw | ForEach-Object { $_.ToString() }) -join "`n"
    return [pscustomobject]@{
        ExitCode = $exitCode
        Text = $text
    }
}

function Invoke-F21Build {
    param(
        [string]$CMake,
        [string]$BuildRoot
    )
    return Invoke-F21Native $CMake @(
        "--build", $BuildRoot, "--config", "Release",
        "--target", "unit_test_DataLoader")
}

function Invoke-F21FocusedTest {
    param(
        [string]$CTest,
        [string]$BuildRoot
    )
    return Invoke-F21Native $CTest @(
        "--test-dir", $BuildRoot, "-C", "Release",
        "-R", "^unit_test_DataLoader$", "--output-on-failure", "-V")
}

function Assert-F21Control {
    param(
        [string]$Label,
        [string]$CMake,
        [string]$CTest,
        [string]$BuildRoot
    )
    $build = Invoke-F21Build $CMake $BuildRoot
    Assert-F21 ($build.ExitCode -eq 0) (
        "$Label control build exited $($build.ExitCode)`n$($build.Text)"
    )
    $test = Invoke-F21FocusedTest $CTest $BuildRoot
    Assert-F21 ($test.ExitCode -eq 0) (
        "$Label control test exited $($test.ExitCode)`n$($test.Text)"
    )
    Assert-F21 (
        $test.Text.Contains(
            "F21_CPP_NAMES canonical=4/4 legacy=4/4 overloads=12/12 " +
            "loader_state=22/22 path_state=16/16 cstring_copy=4/4 " +
            "skips=0 verdict=PASS")
    ) "$Label control omitted the F21 execution marker"
    Assert-F21 (
        $test.Text.Contains("All tests passed (81 assertions in 2 test cases)")
    ) "$Label control omitted the exact Catch2 ledger"
    Write-Output "F21_CONTROL label=$Label build=pass test=pass"
}

$repositoryRoot = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::Combine($PSScriptRoot, ".."))
$buildRoot = Get-F21FullPath $BuildDirectory $repositoryRoot
$dataLoaderPath = Get-F21FullPath "dtwc/DataLoader.hpp" $repositoryRoot
$settingsPath = Get-F21FullPath "dtwc/settings.hpp" $repositoryRoot

Assert-F21InsideRepository $buildRoot $repositoryRoot "build directory"
Assert-F21InsideRepository $dataLoaderPath $repositoryRoot "DataLoader header"
Assert-F21InsideRepository $settingsPath $repositoryRoot "settings header"
Assert-F21 ([System.IO.Directory]::Exists($buildRoot)) (
    "build directory does not exist: $buildRoot")

$cmake = (Get-Command cmake -ErrorAction Stop).Source
$ctest = (Get-Command ctest -ErrorAction Stop).Source
$originals = @{
    $dataLoaderPath = [System.IO.File]::ReadAllBytes($dataLoaderPath)
    $settingsPath = [System.IO.File]::ReadAllBytes($settingsPath)
}
$originalHashes = @{
    $dataLoaderPath = Get-F21Sha256 $dataLoaderPath
    $settingsPath = Get-F21Sha256 $settingsPath
}

$mutations = @(
    @{
        Name = "remove-canonical-start-column"
        Path = $dataLoaderPath
        Expected = "compile"
        Replacements = @(
            [pscustomobject]@{
                Needle = "DataLoader &start_column(int N)"
                Replacement = "DataLoader &f21_removed_start_column(int N)"
                Count = 1
            }
        )
    },
    @{
        Name = "remove-canonical-start-row"
        Path = $dataLoaderPath
        Expected = "compile"
        Replacements = @(
            [pscustomobject]@{
                Needle = "DataLoader &start_row(int N)"
                Replacement = "DataLoader &f21_removed_start_row(int N)"
                Count = 1
            }
        )
    },
    @{
        Name = "remove-canonical-data-path"
        Path = $settingsPath
        Expected = "compile"
        Replacements = @(
            [pscustomobject]@{
                Needle = "inline void set_data_path("
                Replacement = "inline void f21_removed_set_data_path("
                Count = 2
            }
        )
    },
    @{
        Name = "remove-canonical-results-path"
        Path = $settingsPath
        Expected = "compile"
        Replacements = @(
            [pscustomobject]@{
                Needle = "inline void set_results_path("
                Replacement = "inline void f21_removed_set_results_path("
                Count = 2
            }
        )
    },
    @{
        Name = "remove-legacy-start-column"
        Path = $dataLoaderPath
        Expected = "compile"
        Replacements = @(
            [pscustomobject]@{
                Needle = "DataLoader &startColumn(int N)"
                Replacement = "DataLoader &f21_removed_startColumn(int N)"
                Count = 1
            }
        )
    },
    @{
        Name = "remove-legacy-start-row"
        Path = $dataLoaderPath
        Expected = "compile"
        Replacements = @(
            [pscustomobject]@{
                Needle = "DataLoader &startRow(int N)"
                Replacement = "DataLoader &f21_removed_startRow(int N)"
                Count = 1
            }
        )
    },
    @{
        Name = "remove-legacy-data-path"
        Path = $settingsPath
        Expected = "compile"
        Replacements = @(
            [pscustomobject]@{
                Needle = "inline void setDataPath("
                Replacement = "inline void f21_removed_setDataPath("
                Count = 2
            }
        )
    },
    @{
        Name = "remove-legacy-results-path"
        Path = $settingsPath
        Expected = "compile"
        Replacements = @(
            [pscustomobject]@{
                Needle = "inline void setResultsPath("
                Replacement = "inline void f21_removed_setResultsPath("
                Count = 2
            }
        )
    },
    @{
        Name = "swap-start-column-assignment"
        Path = $dataLoaderPath
        Expected = "runtime"
        Replacements = @(
            [pscustomobject]@{
                Needle = "start_col_ = N;"
                Replacement = "start_row_ = N;"
                Count = 1
            }
        )
    },
    @{
        Name = "swap-start-row-assignment"
        Path = $dataLoaderPath
        Expected = "runtime"
        Replacements = @(
            [pscustomobject]@{
                Needle = "start_row_ = N;"
                Replacement = "start_col_ = N;"
                Count = 1
            }
        )
    },
    @{
        Name = "redirect-data-path"
        Path = $settingsPath
        Expected = "runtime"
        Replacements = @(
            [pscustomobject]@{
                Needle = "inline void set_data_path(const fs::path &path) { data = path; }"
                Replacement = "inline void set_data_path(const fs::path &path) { results = path; }"
                Count = 1
            },
            [pscustomobject]@{
                Needle = "inline void set_data_path(const char *path) { data = fs::path(path); }"
                Replacement = "inline void set_data_path(const char *path) { results = fs::path(path); }"
                Count = 1
            }
        )
    },
    @{
        Name = "redirect-results-path"
        Path = $settingsPath
        Expected = "runtime"
        Replacements = @(
            [pscustomobject]@{
                Needle = "inline void set_results_path(const fs::path &path) { results = path; }"
                Replacement = "inline void set_results_path(const fs::path &path) { data = path; }"
                Count = 1
            },
            [pscustomobject]@{
                Needle = "inline void set_results_path(const char *path) { results = fs::path(path); }"
                Replacement = "inline void set_results_path(const char *path) { data = fs::path(path); }"
                Count = 1
            }
        )
    }
)

$killed = 0
$compileKilled = 0
$runtimeKilled = 0

Assert-F21Control "initial" $cmake $ctest $buildRoot

try {
    foreach ($mutation in $mutations) {
        foreach ($path in $originals.Keys) {
            [System.IO.File]::WriteAllBytes($path, $originals[$path])
        }

        $text = ConvertFrom-F21Utf8 $originals[$mutation.Path]
        foreach ($replacement in $mutation.Replacements) {
            $text = Replace-F21Exact `
                $text `
                ([string]$replacement.Needle) `
                ([string]$replacement.Replacement) `
                ([int]$replacement.Count) `
                $mutation.Name
        }
        [System.IO.File]::WriteAllBytes(
            $mutation.Path, (ConvertTo-F21Utf8 $text))

        $build = Invoke-F21Build $cmake $buildRoot
        if ($mutation.Expected -eq "compile") {
            Assert-F21 ($build.ExitCode -ne 0) (
                "$($mutation.Name) survived compilation")
            ++$compileKilled
            ++$killed
            Write-Output (
                "F21_MUTATION name=$($mutation.Name) " +
                "class=compile result=killed")
            continue
        }

        Assert-F21 ($build.ExitCode -eq 0) (
            "$($mutation.Name) was expected to compile but exited " +
            "$($build.ExitCode)`n$($build.Text)")
        $test = Invoke-F21FocusedTest $ctest $buildRoot
        Assert-F21 ($test.ExitCode -ne 0) (
            "$($mutation.Name) survived the runtime gate")
        ++$runtimeKilled
        ++$killed
        Write-Output (
            "F21_MUTATION name=$($mutation.Name) " +
            "class=runtime result=killed")
    }
}
finally {
    foreach ($path in $originals.Keys) {
        [System.IO.File]::WriteAllBytes($path, $originals[$path])
    }
}

foreach ($path in $originals.Keys) {
    Assert-F21 (
        (Get-F21Sha256 $path) -eq $originalHashes[$path]
    ) "source restoration failed for $path"
}

Assert-F21Control "final" $cmake $ctest $buildRoot
Assert-F21 ($killed -eq $mutations.Count) (
    "expected $($mutations.Count) kills, observed $killed")
Assert-F21 ($compileKilled -eq 8) (
    "expected 8 compile kills, observed $compileKilled")
Assert-F21 ($runtimeKilled -eq 4) (
    "expected 4 runtime kills, observed $runtimeKilled")

Write-Output (
    "F21_MUTATIONS controls=2/2 mutations=$($mutations.Count) " +
    "killed=$killed compile_killed=$compileKilled " +
    "runtime_killed=$runtimeKilled survived=0 source_restore=pass " +
    "verdict=PASS")
