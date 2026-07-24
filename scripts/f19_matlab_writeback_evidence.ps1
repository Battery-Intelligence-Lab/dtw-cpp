Set-StrictMode -Version Latest

function Assert-F19Evidence {
    param(
        [Parameter(Mandatory = $true)]
        [bool] $Condition,

        [Parameter(Mandatory = $true)]
        [string] $Message
    )

    if (-not $Condition) {
        throw "F19 MATLAB evidence: $Message"
    }
}

function Resolve-F19RepositoryPath {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [string] $RepositoryRoot
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

function Assert-F19EvidencePathInside {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [string] $Root,

        [Parameter(Mandatory = $true)]
        [string] $Description,

        [switch] $AllowRoot
    )

    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root)
    if (
        $AllowRoot -and
        [string]::Equals(
            $resolvedPath,
            $resolvedRoot,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        return
    }

    $rootPrefix = $resolvedRoot.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    Assert-F19Evidence (
        $resolvedPath.StartsWith(
            $rootPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) "$Description must resolve inside '$resolvedRoot': $resolvedPath"
}

function Resolve-F19ExecutableFile {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    Assert-F19Evidence (
        [System.IO.File]::Exists($Path)
    ) "$Description does not exist: $Path"
    $currentPath = [System.IO.Path]::GetFullPath($Path)
    for ($depth = 0; $depth -lt 8; ++$depth) {
        $item = Get-Item -LiteralPath $currentPath
        if (
            [string]::IsNullOrWhiteSpace([string]$item.LinkType) -or
            $null -eq $item.Target
        ) {
            return $item.FullName
        }
        $targets = @($item.Target)
        Assert-F19Evidence (
            $targets.Count -eq 1
        ) "$Description link has an ambiguous target: $currentPath"
        $target = [string]$targets[0]
        if (-not [System.IO.Path]::IsPathRooted($target)) {
            $target = Join-Path $item.DirectoryName $target
        }
        $currentPath = [System.IO.Path]::GetFullPath($target)
        Assert-F19Evidence (
            [System.IO.File]::Exists($currentPath)
        ) "$Description link target does not exist: $currentPath"
    }
    throw (
        "F19 MATLAB evidence: $Description link resolution exceeded " +
        "eight hops: $Path"
    )
}

function Get-F19Sha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path
    )

    return (
        Get-FileHash -LiteralPath $Path -Algorithm SHA256
    ).Hash.ToUpperInvariant()
}

function Test-F19ByteEquality {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]] $Left,

        [Parameter(Mandatory = $true)]
        [byte[]] $Right
    )

    if ($Left.Length -ne $Right.Length) {
        return $false
    }
    for ($index = 0; $index -lt $Left.Length; ++$index) {
        if ($Left[$index] -ne $Right[$index]) {
            return $false
        }
    }
    return $true
}

function ConvertTo-F19StrictUtf8NoBomBytes {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string] $Text
    )

    $encoding = New-Object System.Text.UTF8Encoding($false, $true)
    return ,([byte[]]$encoding.GetBytes($Text))
}

function ConvertFrom-F19StrictUtf8NoBomBytes {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]] $Bytes,

        [Parameter(Mandatory = $true)]
        [string] $Description,

        [string] $Origin = '<memory>'
    )

    $hasBom = (
        $Bytes.Length -ge 3 -and
        $Bytes[0] -eq 0xEF -and
        $Bytes[1] -eq 0xBB -and
        $Bytes[2] -eq 0xBF
    )
    Assert-F19Evidence (
        -not $hasBom
    ) "$Description must be UTF-8 without a BOM: $Origin"

    $encoding = New-Object System.Text.UTF8Encoding($false, $true)
    try {
        $text = $encoding.GetString($Bytes)
    }
    catch {
        throw (
            "F19 MATLAB evidence: $Description is not strict UTF-8: " +
            "$Origin ($($_.Exception.Message))"
        )
    }
    $roundTrip = [byte[]]$encoding.GetBytes($text)
    Assert-F19Evidence (
        Test-F19ByteEquality $Bytes $roundTrip
    ) "$Description did not round-trip byte-exactly as UTF-8: $Origin"

    return [pscustomobject]@{
        Bytes = $Bytes
        Text = $text
    }
}

function Read-F19StrictUtf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    $bytes = [byte[]][System.IO.File]::ReadAllBytes($Path)
    return ConvertFrom-F19StrictUtf8NoBomBytes (
        $bytes
    ) $Description $Path
}

function Write-F19EvidenceUtf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string] $Content,

        [Parameter(Mandatory = $true)]
        [string] $RepositoryRoot
    )

    Assert-F19EvidencePathInside (
        $Path
    ) $RepositoryRoot 'evidence artifact'
    $parent = [System.IO.Path]::GetDirectoryName(
        [System.IO.Path]::GetFullPath($Path)
    )
    if (-not [System.IO.Directory]::Exists($parent)) {
        [void][System.IO.Directory]::CreateDirectory($parent)
    }
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Get-F19ProfileSpecifications {
    return @(
        [pscustomobject]@{
            Name = 'inherited'
            Route = 'all'
            Executions = 8
            HelperDefinitions = 1
            HelperAssignments = 1
            Calls = [ordered]@{
                fast_pam = 1
                fast_clara = 1
                clarans = 1
                cut_dendrogram = 1
            }
        },
        [pscustomobject]@{
            Name = 'single_fast_pam'
            Route = 'fast_pam'
            Executions = 2
            HelperDefinitions = 1
            HelperAssignments = 1
            Calls = [ordered]@{
                fast_pam = 0
                fast_clara = 1
                clarans = 1
                cut_dendrogram = 1
            }
        },
        [pscustomobject]@{
            Name = 'single_fast_clara'
            Route = 'fast_clara'
            Executions = 2
            HelperDefinitions = 1
            HelperAssignments = 1
            Calls = [ordered]@{
                fast_pam = 1
                fast_clara = 0
                clarans = 1
                cut_dendrogram = 1
            }
        },
        [pscustomobject]@{
            Name = 'single_clarans'
            Route = 'clarans'
            Executions = 2
            HelperDefinitions = 1
            HelperAssignments = 1
            Calls = [ordered]@{
                fast_pam = 1
                fast_clara = 1
                clarans = 0
                cut_dendrogram = 1
            }
        },
        [pscustomobject]@{
            Name = 'single_cut_dendrogram'
            Route = 'cut_dendrogram'
            Executions = 2
            HelperDefinitions = 1
            HelperAssignments = 1
            Calls = [ordered]@{
                fast_pam = 1
                fast_clara = 1
                clarans = 1
                cut_dendrogram = 0
            }
        },
        [pscustomobject]@{
            Name = 'composite'
            Route = 'all'
            Executions = 8
            HelperDefinitions = 0
            HelperAssignments = 0
            Calls = [ordered]@{
                fast_pam = 0
                fast_clara = 0
                clarans = 0
                cut_dendrogram = 0
            }
        }
    )
}

function Get-F19ProfileSpecification {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Profile
    )

    $matches = @(
        Get-F19ProfileSpecifications |
            Where-Object { $_.Name -eq $Profile }
    )
    Assert-F19Evidence (
        $matches.Count -eq 1
    ) "unknown or duplicate fixed profile '$Profile'."
    return $matches[0]
}

function Remove-F19CppComments {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text
    )

    $withoutBlocks = [System.Text.RegularExpressions.Regex]::Replace(
        $Text,
        '(?s)/\*.*?\*/',
        [System.Text.RegularExpressions.MatchEvaluator] {
            param($match)
            return [System.Text.RegularExpressions.Regex]::Replace(
                $match.Value,
                '[^\r\n]',
                ' '
            )
        }
    )
    return [System.Text.RegularExpressions.Regex]::Replace(
        $withoutBlocks,
        '(?m)//[^\r\n]*',
        ''
    )
}

function Get-F19CommandSection {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [string] $StartFunction,

        [Parameter(Mandatory = $true)]
        [string] $EndFunction
    )

    $startPattern = (
        '(?m)^\s*static\s+void\s+' +
        [System.Text.RegularExpressions.Regex]::Escape($StartFunction) +
        '\s*\('
    )
    $endPattern = (
        '(?m)^\s*static\s+void\s+' +
        [System.Text.RegularExpressions.Regex]::Escape($EndFunction) +
        '\s*\('
    )
    $start = [System.Text.RegularExpressions.Regex]::Match(
        $Text,
        $startPattern
    )
    Assert-F19Evidence (
        $start.Success
    ) "source is missing command function '$StartFunction'."
    $endSearchStart = $start.Index + $start.Length
    $end = [System.Text.RegularExpressions.Regex]::Match(
        $Text.Substring($endSearchStart),
        $endPattern
    )
    Assert-F19Evidence (
        $end.Success
    ) (
        "source is missing the '$EndFunction' boundary after " +
        "'$StartFunction'."
    )
    $endIndex = $endSearchStart + $end.Index
    return $Text.Substring($start.Index, $endIndex - $start.Index)
}

function Get-F19ProfileSourceShape {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text
    )

    $semanticText = Remove-F19CppComments $Text
    $boundaries = [ordered]@{
        fast_pam = @('cmd_fast_pam', 'cmd_fast_clara')
        fast_clara = @('cmd_fast_clara', 'cmd_clarans')
        clarans = @('cmd_clarans', 'cmd_build_dendrogram')
        cut_dendrogram = @('cmd_cut_dendrogram', 'cmd_silhouette')
    }
    $callPattern = (
        '(?m)^\s*store_result_in_problem\s*\(\s*prob\s*,\s*result\s*\)' +
        '\s*;\s*$'
    )
    $calls = [ordered]@{}
    foreach ($route in $boundaries.Keys) {
        $names = $boundaries[$route]
        $section = Get-F19CommandSection (
            $semanticText
        ) $names[0] $names[1]
        $calls[$route] = (
            [System.Text.RegularExpressions.Regex]::Matches(
                $section,
                $callPattern
            ).Count
        )
    }

    $helperDefinitions = (
        [System.Text.RegularExpressions.Regex]::Matches(
            $semanticText,
            (
                '(?m)^\s*static\s+void\s+store_result_in_problem\s*' +
                '\(\s*dtwc::Problem\s*&'
            )
        ).Count
    )
    $tokens = (
        [System.Text.RegularExpressions.Regex]::Matches(
            $semanticText,
            '\bstore_result_in_problem\b'
        ).Count
    )
    $setK = (
        [System.Text.RegularExpressions.Regex]::Matches(
            $semanticText,
            '\bprob\.set_n_clusters\s*\(\s*k\s*\)\s*;'
        ).Count
    )
    $setMedoids = (
        [System.Text.RegularExpressions.Regex]::Matches(
            $semanticText,
            (
                '\bprob\.centroids_ind\s*=\s*' +
                'result\.medoid_indices\s*;'
            )
        ).Count
    )
    $setLabels = (
        [System.Text.RegularExpressions.Regex]::Matches(
            $semanticText,
            '\bprob\.clusters_ind\s*=\s*result\.labels\s*;'
        ).Count
    )

    return [pscustomobject]@{
        HelperDefinitions = $helperDefinitions
        HelperSetK = $setK
        HelperSetMedoids = $setMedoids
        HelperSetLabels = $setLabels
        Tokens = $tokens
        Calls = $calls
    }
}

function Assert-F19ProfileSource {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Profile,

        [Parameter(Mandatory = $true)]
        [string] $Text
    )

    $specification = Get-F19ProfileSpecification $Profile
    $shape = Get-F19ProfileSourceShape $Text
    Assert-F19Evidence (
        $shape.HelperDefinitions -eq $specification.HelperDefinitions
    ) (
        "profile '$Profile' helper-definition drift: observed " +
        "$($shape.HelperDefinitions), expected " +
        "$($specification.HelperDefinitions)."
    )
    foreach ($assignment in @(
        'HelperSetK',
        'HelperSetMedoids',
        'HelperSetLabels'
    )) {
        Assert-F19Evidence (
            $shape.$assignment -eq $specification.HelperAssignments
        ) (
            "profile '$Profile' $assignment drift: observed " +
            "$($shape.$assignment), expected " +
            "$($specification.HelperAssignments)."
        )
    }

    $expectedTokens = $specification.HelperDefinitions
    foreach ($route in $specification.Calls.Keys) {
        $observed = $shape.Calls[$route]
        $expected = $specification.Calls[$route]
        Assert-F19Evidence (
            $observed -eq $expected
        ) (
            "profile '$Profile' route '$route' call-count drift: observed " +
            "$observed, expected $expected."
        )
        $expectedTokens += $expected
    }
    Assert-F19Evidence (
        $shape.Tokens -eq $expectedTokens
    ) (
        "profile '$Profile' helper-token drift: observed " +
        "$($shape.Tokens), expected $expectedTokens."
    )
    return $shape
}

function Remove-F19RouteCallLine {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [ValidateSet(
            'fast_pam',
            'fast_clara',
            'clarans',
            'cut_dendrogram'
        )]
        [string] $Route
    )

    $boundaries = @{
        fast_pam = @('cmd_fast_pam', 'cmd_fast_clara')
        fast_clara = @('cmd_fast_clara', 'cmd_clarans')
        clarans = @('cmd_clarans', 'cmd_build_dendrogram')
        cut_dendrogram = @('cmd_cut_dendrogram', 'cmd_silhouette')
    }
    $names = $boundaries[$Route]
    $startPattern = (
        '(?m)^\s*static\s+void\s+' +
        [System.Text.RegularExpressions.Regex]::Escape($names[0]) +
        '\s*\('
    )
    $endPattern = (
        '(?m)^\s*static\s+void\s+' +
        [System.Text.RegularExpressions.Regex]::Escape($names[1]) +
        '\s*\('
    )
    $start = [System.Text.RegularExpressions.Regex]::Match(
        $Text,
        $startPattern
    )
    Assert-F19Evidence (
        $start.Success
    ) "cannot derive '$Route' mutant: start function is absent."
    $endSearchStart = $start.Index + $start.Length
    $end = [System.Text.RegularExpressions.Regex]::Match(
        $Text.Substring($endSearchStart),
        $endPattern
    )
    Assert-F19Evidence (
        $end.Success
    ) "cannot derive '$Route' mutant: end function is absent."
    $endIndex = $endSearchStart + $end.Index
    $section = $Text.Substring($start.Index, $endIndex - $start.Index)
    $linePattern = (
        '(?m)^[ \t]*store_result_in_problem\s*\(\s*prob\s*,\s*result\s*\)' +
        '\s*;[ \t]*(?:\r?\n|$)'
    )
    $matches = [System.Text.RegularExpressions.Regex]::Matches(
        $section,
        $linePattern
    )
    Assert-F19Evidence (
        $matches.Count -eq 1
    ) (
        "cannot derive '$Route' single deletion: observed " +
        "$($matches.Count) exact helper-call lines."
    )
    $match = $matches[0]
    $absoluteIndex = $start.Index + $match.Index
    return $Text.Remove($absoluteIndex, $match.Length)
}

function Remove-F19ResultWritebackHelper {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text
    )

    $startPattern = (
        '(?m)^[ \t]*/// Store clustering result back into Problem ' +
        '\(CRITICAL for scoring functions\)\r?$'
    )
    $endPattern = (
        '(?m)^[ \t]*/// Build a Dendrogram MATLAB struct\r?$'
    )
    $start = [System.Text.RegularExpressions.Regex]::Match(
        $Text,
        $startPattern
    )
    Assert-F19Evidence (
        $start.Success
    ) 'cannot derive composite deletion: helper documentation is absent.'
    $endSearchStart = $start.Index + $start.Length
    $end = [System.Text.RegularExpressions.Regex]::Match(
        $Text.Substring($endSearchStart),
        $endPattern
    )
    Assert-F19Evidence (
        $end.Success
    ) 'cannot derive composite deletion: helper end boundary is absent.'
    $endIndex = $endSearchStart + $end.Index
    $helperRegion = $Text.Substring(
        $start.Index,
        $endIndex - $start.Index
    )
    Assert-F19Evidence (
        (
            [System.Text.RegularExpressions.Regex]::Matches(
                $helperRegion,
                (
                    '(?m)^\s*static\s+void\s+' +
                    'store_result_in_problem\s*\('
                )
            ).Count
        ) -eq 1
    ) (
        'cannot derive composite deletion: helper region does not contain ' +
        'exactly one definition.'
    )
    return $Text.Remove($start.Index, $endIndex - $start.Index)
}

function Get-F19CompositeSource {
    param(
        [Parameter(Mandatory = $true)]
        [string] $InheritedText
    )

    $result = $InheritedText
    foreach ($route in @(
        'fast_pam',
        'fast_clara',
        'clarans',
        'cut_dendrogram'
    )) {
        $result = Remove-F19RouteCallLine $result $route
    }
    return Remove-F19ResultWritebackHelper $result
}

function Assert-F19BuildLog {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [string] $Profile,

        [Parameter(Mandatory = $true)]
        [ValidatePattern('^[0-9A-Fa-f]{64}$')]
        [string] $SourceSha256,

        [Parameter(Mandatory = $true)]
        [string] $ExpectedMexPath
    )

    $sourceSha = $SourceSha256.ToUpperInvariant()
    $expectedMex = [System.IO.Path]::GetFullPath($ExpectedMexPath)
    $expectedMexPortable = $expectedMex.Replace('\', '/')
    $captureMarker = (
        "F19_CAPTURE_BUILD profile=$Profile " +
        "source_sha256=$sourceSha clean_first=1 " +
        "expected_mex_path=$expectedMexPortable"
    )
    $markerCount = (
        [System.Text.RegularExpressions.Regex]::Matches(
            $Text,
            [System.Text.RegularExpressions.Regex]::Escape($captureMarker)
        ).Count
    )
    Assert-F19Evidence (
        $markerCount -eq 1
    ) (
        "profile '$Profile' build log has $markerCount exact capture " +
        'markers; expected one.'
    )

    $compilePattern = (
        '(?im)^(?!F19_)(?:' +
        '[^\r\n]*\bBuilding\s+CXX\s+object\b[^\r\n]*' +
        '\bdtwc_mex\.cpp\b[^\r\n]*|' +
        '[ \t]*dtwc_mex\.cpp[ \t]*)\r?$'
    )
    $compileLines = (
        [System.Text.RegularExpressions.Regex]::Matches(
            $Text,
            $compilePattern
        ).Count
    )
    Assert-F19Evidence (
        $compileLines -ge 1
    ) (
        "profile '$Profile' build log contains no dtwc_mex.cpp compile line."
    )

    $msvcLinkPattern = (
        '(?im)^(?!F19_)[^\r\n]*?->\s*' +
        '(?<artifact>"?[^"\r\n]*dtwc_mex\.mexw64"?)\s*\r?$'
    )
    $ninjaLinkPattern = (
        '(?im)^(?!F19_)[^\r\n]*?\bLinking\s+CXX\s+shared\s+' +
        '(?:library|module)\s+' +
        '(?<artifact>"?[^"\r\n]*dtwc_mex\.mexw64"?)\s*\r?$'
    )
    $linkMatches = @(
        [System.Text.RegularExpressions.Regex]::Matches(
            $Text,
            $msvcLinkPattern
        )
        [System.Text.RegularExpressions.Regex]::Matches(
            $Text,
            $ninjaLinkPattern
        )
    )
    $linkLines = $linkMatches.Count
    Assert-F19Evidence (
        $linkLines -ge 1
    ) (
        "profile '$Profile' build log contains no dtwc_mex.mexw64 link line."
    )

    $buildBinDirectory = [System.IO.Path]::GetDirectoryName($expectedMex)
    $buildDirectory = [System.IO.Path]::GetDirectoryName($buildBinDirectory)
    $boundLinkLines = 0
    $observedArtifacts = @()
    foreach ($linkMatch in $linkMatches) {
        $artifact = $linkMatch.Groups['artifact'].Value.Trim().Trim('"')
        if ([System.IO.Path]::IsPathRooted($artifact)) {
            $resolvedArtifact = [System.IO.Path]::GetFullPath($artifact)
        }
        else {
            $resolvedArtifact = [System.IO.Path]::GetFullPath(
                (Join-Path $buildDirectory $artifact)
            )
        }
        $observedArtifacts += $resolvedArtifact.Replace('\', '/')
        if (
            [string]::Equals(
                $resolvedArtifact,
                $expectedMex,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        ) {
            $boundLinkLines += 1
        }
    }
    Assert-F19Evidence (
        $boundLinkLines -ge 1
    ) (
        "profile '$Profile' build log does not bind the exact expected MEX " +
        "'$expectedMexPortable'; observed=[" +
        [string]::Join(',', $observedArtifacts) + '].'
    )

    return [pscustomobject]@{
        CaptureMarkers = $markerCount
        CompileLines = $compileLines
        LinkLines = $linkLines
        BoundLinkLines = $boundLinkLines
        ExpectedMexPath = $expectedMex
    }
}
