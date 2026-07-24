[CmdletBinding()]
param(
    [ValidateSet('source', 'hpc', 'cuda', 'all')]
    [string] $Mode = 'all',

    [string] $CpuMexDirectory = '',
    [string] $CudaMexDirectory = '',
    [string] $MatlabR2024b = '',
    [string] $MatlabR2025b = '',
    [string] $NsightCompute = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-F18 {
    param(
        [Parameter(Mandatory = $true)]
        [bool] $Condition,

        [Parameter(Mandatory = $true)]
        [string] $Message
    )

    if (-not $Condition) {
        throw "F18 MATLAB routing gate: $Message"
    }
}

function Get-FullPath {
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

function Assert-PathInsideRepository {
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

    Assert-F18 (
        $resolvedPath.StartsWith(
            $rootPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) "$Description must resolve inside the repository: $resolvedPath"
}

function New-F18Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [string] $RepositoryRoot
    )

    Assert-PathInsideRepository $Path $RepositoryRoot 'runtime directory'
    if (-not [System.IO.Directory]::Exists($Path)) {
        [void][System.IO.Directory]::CreateDirectory($Path)
    }
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string] $Content,

        [Parameter(Mandatory = $true)]
        [string] $RepositoryRoot
    )

    Assert-PathInsideRepository $Path $RepositoryRoot 'runtime artifact'
    $parent = [System.IO.Path]::GetDirectoryName(
        [System.IO.Path]::GetFullPath($Path)
    )
    New-F18Directory $parent $RepositoryRoot
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function ConvertTo-MatlabLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Value
    )

    $portable = $Value.Replace('\', '/').Replace("'", "''")
    return "'$portable'"
}

function Get-LiteralCount {
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

function Get-UniqueLiteralIndex {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [string] $Literal,

        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    $count = Get-LiteralCount $Text $Literal
    Assert-F18 ($count -eq 1) (
        "$Description must occur exactly once; observed $count occurrence(s) " +
        "of '$Literal'."
    )
    return $Text.IndexOf($Literal, [System.StringComparison]::Ordinal)
}

function Assert-LiteralPresent {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [string] $Literal,

        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    Assert-F18 ($Text.Contains($Literal)) (
        "$Description is absent; expected literal '$Literal'."
    )
}

function Assert-RegexPresent {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [string] $Pattern,

        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    Assert-F18 (
        [System.Text.RegularExpressions.Regex]::IsMatch(
            $Text,
            $Pattern,
            [System.Text.RegularExpressions.RegexOptions]::Singleline
        )
    ) "$Description is absent."
}

function Assert-OrderedIndices {
    param(
        [Parameter(Mandatory = $true)]
        [int[]] $Indices,

        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    for ($i = 1; $i -lt $Indices.Count; ++$i) {
        Assert-F18 ($Indices[$i - 1] -lt $Indices[$i]) (
            "$Description is out of order at positions " +
            "$($Indices[$i - 1]) and $($Indices[$i])."
        )
    }
}

function Get-SourceSection {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [string] $StartLiteral,

        [Parameter(Mandatory = $true)]
        [string] $EndLiteral,

        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    $start = Get-UniqueLiteralIndex $Text $StartLiteral "$Description start"
    $end = $Text.IndexOf(
        $EndLiteral,
        $start + $StartLiteral.Length,
        [System.StringComparison]::Ordinal
    )
    Assert-F18 ($end -gt $start) "$Description end marker is absent or unordered."
    return $Text.Substring($start, $end - $start)
}

function Get-CppFunctionSection {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [string] $FunctionPrefix,

        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    $start = Get-UniqueLiteralIndex $Text $FunctionPrefix "$Description definition"
    $next = $Text.IndexOf(
        "`nstatic ",
        $start + $FunctionPrefix.Length,
        [System.StringComparison]::Ordinal
    )
    Assert-F18 ($next -gt $start) "$Description has no following function boundary."
    return $Text.Substring($start, $next - $start)
}

function Invoke-CapturedNative {
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
        [string] $RepositoryRoot
    )

    Assert-F18 ([System.IO.File]::Exists($FilePath)) (
        "native executable does not exist: $FilePath"
    )
    Assert-PathInsideRepository $LogPath $RepositoryRoot 'native log'

    $savedEnvironment = @{}
    foreach ($name in $Environment.Keys) {
        $savedEnvironment[$name] = [System.Environment]::GetEnvironmentVariable(
            [string]$name,
            [System.EnvironmentVariableTarget]::Process
        )
    }

    $hadNativePreference = Test-Path variable:PSNativeCommandUseErrorActionPreference
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
        $outputLines = @(& $FilePath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
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
    Write-Utf8NoBom $LogPath $text $RepositoryRoot
    if ($text.Length -gt 0) {
        Write-Host -NoNewline $text
    }

    return [pscustomobject]@{
        ExitCode = [int]$exitCode
        Text = $text
    }
}

function Assert-NoProfilerCaps {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    foreach ($argument in $Arguments) {
        Assert-F18 (
            $argument -notmatch '(?i)^--launch-(?:count|skip)(?:=|$)'
        ) "Nsight command must be uncapped, but contains '$argument'."
    }
}

function Get-ProfilerInvocationRows {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text
    )

    $rows = @()
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -notmatch '\bInvocations\s+\d+\b') {
            continue
        }

        $match = [System.Text.RegularExpressions.Regex]::Match(
            $line,
            '(?i)\bdtw_[A-Za-z0-9_]*kernel\b.*\bDevice\s+(\d+)\b.*\bCC\s+([0-9]+\.[0-9]+)\b.*\bInvocations\s+(\d+)\b'
        )
        Assert-F18 $match.Success (
            "profiled invocation row is not a production dtw_*kernel row: $line"
        )

        $rows += [pscustomobject]@{
            Line = $line
            Device = [int]$match.Groups[1].Value
            ComputeCapability = $match.Groups[2].Value
            Invocations = [int]$match.Groups[3].Value
        }
    }
    return @($rows)
}

$script:RepositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..')
)
$script:EstimatorPath = Join-Path $script:RepositoryRoot (
    'bindings\matlab\+dtwc\DTWClustering.m'
)
$script:MexSourcePath = Join-Path $script:RepositoryRoot (
    'bindings\matlab\dtwc_mex.cpp'
)

if ([string]::IsNullOrWhiteSpace($CpuMexDirectory)) {
    $CpuMexDirectory = Join-Path $script:RepositoryRoot (
        'build\mex-verify-msvc\bin'
    )
}
else {
    $CpuMexDirectory = Get-FullPath $CpuMexDirectory $script:RepositoryRoot
}

if ([string]::IsNullOrWhiteSpace($CudaMexDirectory)) {
    $CudaMexDirectory = Join-Path $script:RepositoryRoot (
        'build\mex-cuda-f18\bin'
    )
}
else {
    $CudaMexDirectory = Get-FullPath $CudaMexDirectory $script:RepositoryRoot
}

if ([string]::IsNullOrWhiteSpace($MatlabR2024b)) {
    $MatlabR2024b = 'C:\Program Files\MATLAB\R2024b\bin\matlab.exe'
}
else {
    $MatlabR2024b = Get-FullPath $MatlabR2024b $script:RepositoryRoot
}

if ([string]::IsNullOrWhiteSpace($MatlabR2025b)) {
    $MatlabR2025b = 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe'
}
else {
    $MatlabR2025b = Get-FullPath $MatlabR2025b $script:RepositoryRoot
}

if ([string]::IsNullOrWhiteSpace($NsightCompute)) {
    $NsightCompute = (
        'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\' +
        'target\windows-desktop-win7-x64\ncu.exe'
    )
}
else {
    $NsightCompute = Get-FullPath $NsightCompute $script:RepositoryRoot
}

$CpuMexDirectory = [System.IO.Path]::GetFullPath($CpuMexDirectory)
$CudaMexDirectory = [System.IO.Path]::GetFullPath($CudaMexDirectory)
$script:CpuMexPath = Join-Path $CpuMexDirectory 'dtwc_mex.mexw64'
$script:CudaMexPath = Join-Path $CudaMexDirectory 'dtwc_mex.mexw64'

Assert-PathInsideRepository (
    $CpuMexDirectory
) $script:RepositoryRoot 'ordinary MEX directory'
Assert-PathInsideRepository (
    $CudaMexDirectory
) $script:RepositoryRoot 'CUDA MEX directory'

function Test-F18Source {
    Assert-F18 ([System.IO.File]::Exists($script:EstimatorPath)) (
        "estimator source does not exist: $script:EstimatorPath"
    )
    Assert-F18 ([System.IO.File]::Exists($script:MexSourcePath)) (
        "MEX source does not exist: $script:MexSourcePath"
    )

    $matlab = [System.IO.File]::ReadAllText($script:EstimatorPath)
    $mex = [System.IO.File]::ReadAllText($script:MexSourcePath)

    $metricAnchor = Get-UniqueLiteralIndex $matlab (
        '% F18 metric validation must precede input, device, and distance effects.'
    ) 'metric-order anchor'
    $metricCall = Get-UniqueLiteralIndex $matlab (
        'metric = obj.normalized_metric();'
    ) 'normalized metric call'
    $requestedDevice = Get-UniqueLiteralIndex $matlab (
        'requestedDevice = lower(strtrim(obj.Device));'
    ) 'normalized requested device'
    $explicitAnchor = Get-UniqueLiteralIndex $matlab (
        '% F18 explicit-HPC guard must precede dtwc.device.'
    ) 'explicit-HPC anchor'
    $explicitCheck = Get-UniqueLiteralIndex $matlab (
        "if strcmp(requestedDevice, 'hpc')"
    ) 'explicit-HPC check'
    $deviceMutation = Get-UniqueLiteralIndex $matlab (
        'dtwc.device(requestedDevice);'
    ) 'explicit local-device delegation'
    $effectiveDevice = Get-UniqueLiteralIndex $matlab (
        'effectiveDevice = lower(strtrim(dtwc.device()));'
    ) 'normalized effective device'
    $activeAnchor = Get-UniqueLiteralIndex $matlab (
        '% F18 active-HPC guard must precede input and local work.'
    ) 'active-HPC anchor'
    $activeCheck = Get-UniqueLiteralIndex $matlab (
        "if strcmp(effectiveDevice, 'hpc')"
    ) 'active-HPC check'
    $inputMaterialisation = Get-UniqueLiteralIndex $matlab (
        "validateattributes(X, {'numeric'}, {'2d', 'nonempty'}, 'fit', 'X');"
    ) 'fit input validation'
    $producerAnchor = Get-UniqueLiteralIndex $matlab (
        '% F18 routed matrix is produced exactly once before the restart loop.'
    ) 'routed-matrix anchor'
    $producer = Get-UniqueLiteralIndex $matlab (
        "distanceMatrix = dtwc_mex('DTWClustering_compute_distance_matrix', double(X), double(obj.Band), metric);"
    ) 'routed matrix production'
    $restart = Get-UniqueLiteralIndex $matlab (
        'for rep = 1:obj.NInit'
    ) 'restart loop'
    $injectAnchor = Get-UniqueLiteralIndex $matlab (
        '% F18 inject only after every distance-semantic setter.'
    ) 'matrix-injection anchor'
    $inject = Get-UniqueLiteralIndex $matlab (
        'prob.set_distance_matrix(distanceMatrix);'
    ) 'matrix injection'
    $fastPam = Get-UniqueLiteralIndex $matlab (
        'result = dtwc.fast_pam(prob, obj.NClusters, ...'
    ) 'FastPAM call'

    Assert-OrderedIndices @(
        $metricAnchor,
        $metricCall,
        $requestedDevice,
        $explicitAnchor,
        $explicitCheck,
        $deviceMutation,
        $effectiveDevice,
        $activeAnchor,
        $activeCheck,
        $inputMaterialisation,
        $producerAnchor,
        $producer,
        $restart,
        $injectAnchor,
        $inject,
        $fastPam
    ) 'estimator metric/HPC/production/restart/injection contract'

    Assert-LiteralPresent $matlab (
        "metric = lower(strtrim(obj.Metric));"
    ) 'case-normalized metric'
    Assert-LiteralPresent $matlab "'l1'" 'L1 metric token'
    Assert-LiteralPresent $matlab (
        "'squared_euclidean'"
    ) 'SquaredL2 metric token'
    Assert-LiteralPresent $matlab (
        "Unknown Metric '%s'. Expected one of: l1, squared_euclidean."
    ) 'unknown-metric diagnostic'
    Assert-F18 (
        (Get-LiteralCount $matlab (
            "MATLAB DTWClustering does not implement device='hpc'; no SSH/SLURM transport was attempted."
        )) -ge 1
    ) 'registered estimator HPC diagnostic is absent.'
    Assert-F18 ($explicitAnchor -lt $deviceMutation) (
        'explicit-HPC rejection follows dtwc.device.'
    )
    Assert-F18 ($activeAnchor -lt $inputMaterialisation) (
        'active-HPC rejection follows input materialisation.'
    )
    Assert-F18 ($explicitAnchor -lt $producer -and $activeAnchor -lt $producer) (
        'an HPC rejection follows matrix production.'
    )
    Assert-F18 ($explicitAnchor -lt $restart -and $activeAnchor -lt $restart) (
        'an HPC rejection follows restart entry.'
    )

    Assert-RegexPresent $matlab (
        "if\s+strcmp\(effectiveDevice,\s*'cpu'\)\s*&&\s*strcmp\(metric,\s*'l1'\)"
    ) 'explicit lazy CPU-L1 route'
    Assert-F18 ($producer -lt $restart) (
        'routed matrix production is inside or after the restart loop.'
    )
    $loopBody = $matlab.Substring($restart)
    Assert-F18 (
        -not $loopBody.Contains(
            "dtwc_mex('DTWClustering_compute_distance_matrix'"
        )
    ) 'routed matrix production appears inside the restart loop.'

    $setData = $matlab.IndexOf(
        'prob.set_data(double(X));',
        $restart,
        [System.StringComparison]::Ordinal
    )
    $setBand = $matlab.IndexOf(
        'prob.Band = obj.Band;',
        $restart,
        [System.StringComparison]::Ordinal
    )
    $setVariant = $matlab.LastIndexOf(
        'prob.set_variant(',
        $inject,
        [System.StringComparison]::Ordinal
    )
    $setMissing = $matlab.LastIndexOf(
        'prob.set_missing_strategy(',
        $inject,
        [System.StringComparison]::Ordinal
    )
    Assert-F18 (
        $setData -gt $restart -and
        $setBand -gt $setData -and
        $setVariant -gt $setBand -and
        $setMissing -gt $setVariant -and
        $injectAnchor -gt $setMissing -and
        $inject -gt $injectAnchor -and
        $fastPam -gt $inject
    ) 'matrix injection does not follow every distance-semantic setter.'

    $mexFunction = Get-CppFunctionSection $mex `
        'static void cmd_DTWClustering_compute_distance_matrix' `
        'DTWClustering routed MEX command'

    $boundsAnchor = Get-UniqueLiteralIndex $mexFunction (
        '// F18 checked cardinalities precede every producer/allocation.'
    ) 'checked-cardinality anchor'
    $cudaAnchor = Get-UniqueLiteralIndex $mexFunction (
        '// F18 CUDA producer: no CPU fallback.'
    ) 'CUDA producer anchor'
    $metalAnchor = Get-UniqueLiteralIndex $mexFunction (
        '// F18 Metal producer: no CPU fallback.'
    ) 'Metal producer anchor'
    $resultAnchor = Get-UniqueLiteralIndex $mexFunction (
        '// F18 validate producer result before MATLAB allocation/copy.'
    ) 'producer-result anchor'
    $allocation = $mexFunction.IndexOf(
        'row_major_matrix_to_mx(',
        [System.StringComparison]::Ordinal
    )
    Assert-F18 ($allocation -ge 0) 'validated matrix is not passed to the copy helper.'

    Assert-OrderedIndices @(
        $boundsAnchor,
        $cudaAnchor,
        $metalAnchor,
        $resultAnchor,
        $allocation
    ) 'MEX bounds/backend/result/allocation/copy contract'

    Assert-LiteralPresent $mexFunction (
        'checked_product(N, N, "DTWClustering matrix cardinality")'
    ) 'checked N*N arithmetic call'
    Assert-LiteralPresent $mexFunction (
        'checked_pair_count(N)'
    ) 'checked pair-count arithmetic call'
    Assert-LiteralPresent $mexFunction (
        'validate_backend_index_ranges('
    ) 'backend int-narrowing guard call'

    $checkedProduct = Get-CppFunctionSection $mex `
        'static std::size_t checked_product' `
        'checked-product helper'
    Assert-RegexPresent $checkedProduct (
        'rhs\s*>\s*std::numeric_limits<std::size_t>::max\(\)\s*/\s*lhs'
    ) 'checked multiplication overflow condition'
    $pairCount = Get-CppFunctionSection $mex `
        'static std::size_t checked_pair_count' `
        'checked-pair-count helper'
    Assert-LiteralPresent $pairCount (
        '"DTWClustering pair count"'
    ) 'checked pair-count label'
    Assert-RegexPresent $pairCount (
        'checked_product\s*\('
    ) 'divided-factor pair multiplication'
    $indexRanges = Get-CppFunctionSection $mex `
        'static void validate_backend_index_ranges' `
        'backend-index-range helper'
    Assert-LiteralPresent $indexRanges (
        'std::numeric_limits<int>::max()'
    ) 'backend int narrowing limit'

    $cudaSection = Get-SourceSection $mexFunction (
        '// F18 CUDA producer: no CPU fallback.'
    ) '// F18 Metal producer: no CPU fallback.' 'CUDA branch'
    $metalSection = Get-SourceSection $mexFunction (
        '// F18 Metal producer: no CPU fallback.'
    ) '// F18 validate producer result before MATLAB allocation/copy.' (
        'Metal branch'
    )
    $resultSection = $mexFunction.Substring($resultAnchor)

    Assert-LiteralPresent $cudaSection '#ifdef DTWC_HAS_CUDA' (
        'CUDA optional-backend compile guard'
    )
    Assert-LiteralPresent $cudaSection (
        'dtwc::cuda::cuda_available()'
    ) 'CUDA runtime availability guard'
    Assert-LiteralPresent $cudaSection (
        'opts.device_id = dtwc::env().device_index();'
    ) 'CUDA Env ordinal routing'
    Assert-LiteralPresent $cudaSection (
        'opts.use_squared_l2 = (metric == dtwc::core::MetricType::SquaredL2);'
    ) 'CUDA normalized metric routing'
    Assert-LiteralPresent $cudaSection (
        'dtwc::cuda::compute_distance_matrix_cuda(series, opts)'
    ) 'CUDA matrix producer'

    Assert-RegexPresent $metalSection (
        '#(?:if|elif)\s+defined\(DTWC_HAS_METAL\)|#ifdef\s+DTWC_HAS_METAL'
    ) (
        'Metal optional-backend compile guard'
    )
    Assert-LiteralPresent $metalSection (
        'dtwc::metal::metal_available()'
    ) 'Metal runtime availability guard'
    Assert-LiteralPresent $metalSection (
        'dtwc::env().device_index() != 0'
    ) 'Metal nonzero-ordinal rejection'
    Assert-LiteralPresent $metalSection (
        'opts.use_squared_l2 = (metric == dtwc::core::MetricType::SquaredL2);'
    ) 'Metal normalized metric routing'
    Assert-LiteralPresent $metalSection (
        'dtwc::metal::compute_distance_matrix_metal(series, opts)'
    ) 'Metal matrix producer'
    Assert-LiteralPresent $mexFunction '#else' 'GPU-not-compiled branch'
    Assert-LiteralPresent $mexFunction (
        'throw dtwc::DeviceError'
    ) 'GPU fail-closed error'

    foreach ($backendSection in @($cudaSection, $metalSection)) {
        Assert-F18 (
            $backendSection -notmatch (
                '(?i)fillDistanceMatrix|BruteForce|' +
                'compute_distance_matrix_cpu|dtwc::Problem'
            )
        ) 'an optional-backend branch contains a CPU fallback.'
    }

    Assert-LiteralPresent $resultSection (
        'validate_backend_result('
    ) 'producer result validator call'
    $resultValidator = Get-CppFunctionSection $mex `
        'static void validate_backend_result' `
        'producer-result validator'
    Assert-LiteralPresent $resultValidator (
        'result.n != expected_n'
    ) 'producer result-dimension validation'
    Assert-LiteralPresent $resultValidator (
        'result.matrix.size() != matrix_elements'
    ) 'producer matrix-storage validation'
    Assert-LiteralPresent $resultValidator (
        'result.pairs_pruned != 0'
    ) 'producer no-pruning validation'
    Assert-RegexPresent $resultValidator (
        'expected_n\s*>\s*1.*result\.pairs_computed\s*!=\s*expected_pairs'
    ) 'producer full-pair validation'
    Assert-RegexPresent $resultValidator (
        'expected_n\s*>\s*1.*result\.kernel_used\.empty\(\).*result\.kernel_used\s*==\s*"none"'
    ) 'producer kernel-name validation'
    $copyHelper = Get-CppFunctionSection $mex `
        'static mxArray *row_major_matrix_to_mx' `
        'row-major MATLAB copy helper'
    Assert-LiteralPresent $copyHelper (
        'out[i + j * n] = matrix[i * n + j];'
    ) 'row-major to MATLAB column-major copy'

    Assert-LiteralPresent $mex (
        'else if (cmd == "DTWClustering_compute_distance_matrix")'
    ) 'hidden estimator command registration'
    Assert-F18 (
        (Get-LiteralCount $mex (
            'cmd_DTWClustering_compute_distance_matrix'
        )) -eq 2
    ) 'hidden estimator command must have one definition and one registration.'

    Write-Output (
        'F18_MATLAB_SOURCE subject=DTWClustering+dtwc_mex ' +
        'metric_before_effects=1/1 hpc_before_compute=2/2 ' +
        'cpu_l1_lazy=1/1 cpu_squared_outside=1/1 ' +
        'inject_after_setters=1/1 checked_bounds=1/1 ' +
        'cuda_guard=1/1 cuda_metric=1/1 cuda_ordinal=1/1 ' +
        'cuda_result=1/1 metal_guard=1/1 metal_metric=1/1 ' +
        'metal_ordinal=1/1 metal_result=1/1 no_cpu_fallback=2/2 ' +
        'matrix_copy=1/1 skips=0'
    )
}

function Test-F18Hpc {
    Assert-F18 ([System.IO.File]::Exists($script:CpuMexPath)) (
        "ordinary MEX does not exist: $script:CpuMexPath"
    )
    Assert-F18 ([System.IO.File]::Exists($MatlabR2024b)) (
        "MATLAB R2024b executable does not exist: $MatlabR2024b"
    )

    $runtimeRoot = Join-Path $script:RepositoryRoot 'build\f18-hpc-poison'
    $fakeRepository = Join-Path $runtimeRoot 'fake-repository'
    $shimDirectory = Join-Path $runtimeRoot 'path-first'
    $preferences = Join-Path $runtimeRoot 'matlab-pref'
    $temporary = Join-Path $runtimeRoot 'temp'
    foreach ($directory in @(
        $runtimeRoot,
        $fakeRepository,
        $shimDirectory,
        $preferences,
        $temporary
    )) {
        New-F18Directory $directory $script:RepositoryRoot
    }

    $sshLog = Join-Path $runtimeRoot 'ssh-calls.txt'
    $matlabLog = Join-Path $runtimeRoot 'matlab-hpc.log'
    $batchPath = Join-Path $runtimeRoot 'f18_hpc_poison.m'
    foreach ($file in @($sshLog, $matlabLog)) {
        if ([System.IO.File]::Exists($file)) {
            Remove-Item -LiteralPath $file -Force
        }
    }

    $fakeEnv = @'
SLURM_HOST=f18.invalid
SLURM_USER=f18_user
SLURM_REMOTE_BASE=/f18/never
'@
    Write-Utf8NoBom (
        (Join-Path $fakeRepository '.env')
    ) $fakeEnv $script:RepositoryRoot

    $shimPath = Join-Path $shimDirectory 'ssh.cmd'
    $shim = @'
@echo off
if not defined F18_SSH_LOG exit /b 91
>>"%F18_SSH_LOG%" echo %*
exit /b 0
'@
    Write-Utf8NoBom $shimPath $shim $script:RepositoryRoot

    $originalPath = [System.Environment]::GetEnvironmentVariable(
        'PATH',
        [System.EnvironmentVariableTarget]::Process
    )
    $pathWithShim = "$shimDirectory;$originalPath"
    $originalPathExt = [System.Environment]::GetEnvironmentVariable(
        'PATHEXT',
        [System.EnvironmentVariableTarget]::Process
    )
    $pathExt = $originalPathExt
    if ([string]::IsNullOrWhiteSpace($pathExt)) {
        $pathExt = '.COM;.EXE;.BAT;.CMD'
    }
    elseif ($pathExt -notmatch '(?i)(?:^|;)\.CMD(?:;|$)') {
        $pathExt = ".CMD;$pathExt"
    }

    $whereExe = Join-Path (
        [System.Environment]::GetFolderPath(
            [System.Environment+SpecialFolder]::System
        )
    ) 'where.exe'
    Assert-F18 ([System.IO.File]::Exists($whereExe)) (
        "where.exe does not exist: $whereExe"
    )

    $savedPath = $env:PATH
    $savedPathExt = $env:PATHEXT
    try {
        $env:PATH = $pathWithShim
        $env:PATHEXT = $pathExt
        $whereOutput = @(& $whereExe ssh 2>&1)
        $whereExit = $LASTEXITCODE
    }
    finally {
        $env:PATH = $savedPath
        $env:PATHEXT = $savedPathExt
    }
    Assert-F18 ($whereExit -eq 0) (
        "where.exe ssh failed: $([string]::Join(' | ', $whereOutput))"
    )
    Assert-F18 ($whereOutput.Count -ge 1) 'where.exe ssh returned no path.'
    $whereFirst = [System.IO.Path]::GetFullPath($whereOutput[0].ToString().Trim())
    Assert-F18 (
        [string]::Equals(
            $whereFirst,
            [System.IO.Path]::GetFullPath($shimPath),
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) (
        "repo-local poison shim is not the first where.exe ssh result; " +
        "observed '$whereFirst'."
    )

    $repoLiteral = ConvertTo-MatlabLiteral $script:RepositoryRoot
    $bindingsLiteral = ConvertTo-MatlabLiteral (
        (Join-Path $script:RepositoryRoot 'bindings\matlab')
    )
    $mexDirectoryLiteral = ConvertTo-MatlabLiteral $CpuMexDirectory
    $mexPathLiteral = ConvertTo-MatlabLiteral $script:CpuMexPath
    $sshLogLiteral = ConvertTo-MatlabLiteral $sshLog

    $batch = @"
restoredefaultpath;
cd($repoLiteral);
addpath($bindingsLiteral);
addpath($mexDirectoryLiteral);
clear dtwc_mex;
mexPaths=which('dtwc_mex','-all');
if ischar(mexPaths), mexPaths={mexPaths}; end
for i=1:numel(mexPaths), fprintf('MEX_ALL_%d=%s\n',i,mexPaths{i}); end
assert(numel(mexPaths)==1);
assert(strcmp(strrep(mexPaths{1},char(92),'/'),$mexPathLiteral));
X=[0 1;3 8;5 2;6 4];
sshLog=$sshLogLiteral;
expectedMessage='MATLAB DTWClustering does not implement device=''hpc''; no SSH/SLURM transport was attempted.';
dtwc.device('cpu');
explicitModel=dtwc.DTWClustering('NClusters',2,'Metric','l1','Device',' HpC ','NInit',2);
explicitRejected=false;
try
    explicitModel=explicitModel.fit(X);
catch ME
    explicitRejected=strcmp(ME.identifier,'dtwc:deviceError') && strcmp(ME.message,expectedMessage);
end
assert(explicitRejected);
assert(isempty(explicitModel.Labels) && isempty(explicitModel.MedoidIndices) && isnan(explicitModel.TotalCost));
assert(strcmp(dtwc.device(),'cpu'));
assert(~isfile(sshLog) || isempty(fileread(sshLog)));
selected=dtwc.device('hpc');
assert(strcmp(selected,'hpc') && strcmp(dtwc.device(),'hpc'));
assert(isfile(sshLog));
expectedSsh='-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new f18_user@f18.invalid true';
assert(strcmp(strtrim(fileread(sshLog)),expectedSsh));
activeModel=dtwc.DTWClustering('NClusters',2,'Metric','l1','Device','','NInit',2);
activeRejected=false;
try
    activeModel=activeModel.fit(X);
catch ME
    activeRejected=strcmp(ME.identifier,'dtwc:deviceError') && strcmp(ME.message,expectedMessage);
end
assert(activeRejected);
assert(isempty(activeModel.Labels) && isempty(activeModel.MedoidIndices) && isnan(activeModel.TotalCost));
assert(strcmp(dtwc.device(),'hpc'));
assert(strcmp(strtrim(fileread(sshLog)),expectedSsh));
fprintf('F18_MATLAB_HPC explicit_normalized=1/1 explicit_rejected=1/1 explicit_no_ssh=1/1 explicit_device_unchanged=1/1 active_selected=1/1 active_rejected=1/1 active_no_extra_ssh=1/1 no_result=2/2 skips=0\n');
"@
    Write-Utf8NoBom $batchPath $batch $script:RepositoryRoot

    $runExpression = 'run(' + (
        ConvertTo-MatlabLiteral $batchPath
    ) + ');'
    $environment = @{
        'PATH' = $pathWithShim
        'PATHEXT' = $pathExt
        'DTWC_REPO_ROOT' = $fakeRepository
        'F18_SSH_LOG' = $sshLog
        'MATLAB_PREFDIR' = $preferences
        'TEMP' = $temporary
        'TMP' = $temporary
    }
    $result = Invoke-CapturedNative $MatlabR2024b @(
        '-batch',
        $runExpression
    ) $matlabLog $environment $script:RepositoryRoot

    Assert-F18 ($result.ExitCode -eq 0) (
        "HPC poison MATLAB process exited $($result.ExitCode); see $matlabLog"
    )
    $marker = (
        'F18_MATLAB_HPC explicit_normalized=1/1 explicit_rejected=1/1 ' +
        'explicit_no_ssh=1/1 explicit_device_unchanged=1/1 ' +
        'active_selected=1/1 active_rejected=1/1 ' +
        'active_no_extra_ssh=1/1 no_result=2/2 skips=0'
    )
    Assert-F18 ((Get-LiteralCount $result.Text $marker) -eq 1) (
        'HPC process did not print its exact registered marker once.'
    )
    Assert-F18 ([System.IO.File]::Exists($sshLog)) (
        'poison shim did not create its call log.'
    )
    $sshCalls = [System.IO.File]::ReadAllLines($sshLog)
    Assert-F18 ($sshCalls.Count -eq 1) (
        "expected exactly one poison-shim call; observed $($sshCalls.Count)."
    )
    Assert-F18 (
        $sshCalls[0].Trim() -eq (
            '-o BatchMode=yes -o ConnectTimeout=10 ' +
            '-o StrictHostKeyChecking=accept-new ' +
            'f18_user@f18.invalid true'
        )
    ) "poison shim recorded unexpected arguments: '$($sshCalls[0])'."
}

function New-F18CudaBatch {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('valid', 'no-kernel')]
        [string] $Profile,

        [Parameter(Mandatory = $true)]
        [string] $Release,

        [Parameter(Mandatory = $true)]
        [string] $Path
    )

    $repoLiteral = ConvertTo-MatlabLiteral $script:RepositoryRoot
    $bindingsLiteral = ConvertTo-MatlabLiteral (
        (Join-Path $script:RepositoryRoot 'bindings\matlab')
    )
    $mexDirectoryLiteral = ConvertTo-MatlabLiteral $CudaMexDirectory
    $mexPathLiteral = ConvertTo-MatlabLiteral $script:CudaMexPath

    $preamble = @"
restoredefaultpath;
cd($repoLiteral);
addpath($bindingsLiteral);
addpath($mexDirectoryLiteral);
clear dtwc_mex;
mexPaths=which('dtwc_mex','-all');
if ischar(mexPaths), mexPaths={mexPaths}; end
for i=1:numel(mexPaths), fprintf('MEX_ALL_%d=%s\n',i,mexPaths{i}); end
assert(numel(mexPaths)==1);
assert(strcmp(strrep(mexPaths{1},char(92),'/'),$mexPathLiteral));
fprintf('F18_MEX_PATH version=$Release profile=$Profile path=%s\n',mexPaths{1});
X=[0 1;3 8;5 2;6 4];
"@

    if ($Profile -eq 'valid') {
        $body = @"
dtwc.device('gpu');
l1=dtwc.DTWClustering('NClusters',2,'Metric','l1','Device','','NInit',2);
l1=l1.fit(X);
assert(isequal(l1.Labels,int32([1 2 1 1])));
assert(isequal(l1.MedoidIndices,int32([3 2])));
assert(l1.TotalCost==9);
assert(strcmp(dtwc.device(),'gpu'));
dtwc.device('cpu');
squared=dtwc.DTWClustering('NClusters',2,'Metric','SQUARED_EUCLIDEAN','Device','gpu:0','NInit',2);
squared=squared.fit(X);
assert(isequal(squared.Labels,int32([2 1 1 1])));
assert(isequal(squared.MedoidIndices,int32([4 1])));
assert(squared.TotalCost==30);
assert(strcmp(dtwc.device(),'gpu'));
fprintf('F18_CUDA_VALID version=$Release l1_labels=[1 2 1 1] l1_medoids=[3 2] l1_cost=9 squared_labels=[2 1 1 1] squared_medoids=[4 1] squared_cost=30 active_l1=gpu explicit_squared=gpu ninit=2\n');
"@
    }
    else {
        $body = @"
dtwc.device('gpu');
cpu=dtwc.DTWClustering('NClusters',2,'Metric','l1','Device','cpu','NInit',2);
cpu=cpu.fit(X);
assert(isequal(cpu.Labels,int32([1 2 1 1])));
assert(isequal(cpu.MedoidIndices,int32([3 2])));
assert(cpu.TotalCost==9);
assert(strcmp(dtwc.device(),'cpu'));
variantRejected=false;
try
    badVariant=dtwc.DTWClustering('NClusters',2,'Metric','l1','Variant','wdtw','Device','gpu','NInit',2);
    badVariant=badVariant.fit(X);
catch ME
    variantRejected=strcmp(ME.identifier,'dtwc:invalidArgument');
end
assert(variantRejected && strcmp(dtwc.device(),'cpu'));
missingRejected=false;
try
    badMissing=dtwc.DTWClustering('NClusters',2,'Metric','l1','MissingStrategy','zero_cost','Device','gpu','NInit',2);
    badMissing=badMissing.fit(X);
catch ME
    missingRejected=strcmp(ME.identifier,'dtwc:invalidArgument');
end
assert(missingRejected && strcmp(dtwc.device(),'cpu'));
fprintf('F18_CUDA_NO_KERNEL version=$Release cpu_labels=[1 2 1 1] cpu_medoids=[3 2] cpu_cost=9 cpu_override=1/1 variant_rejected=1/1 missing_rejected=1/1 device_unchanged=2/2\n');
"@
    }

    $batch = $preamble + $body
    Assert-F18 (
        $batch -notmatch 'dtwc\.test\.gpu|dtwc\.Problem'
    ) 'CUDA estimator profile contains a forbidden direct GPU probe/Problem.'
    Write-Utf8NoBom $Path $batch $script:RepositoryRoot
}

function Test-F18Cuda {
    Assert-F18 ([System.IO.File]::Exists($script:CudaMexPath)) (
        "CUDA MEX does not exist: $script:CudaMexPath"
    )
    Assert-F18 ([System.IO.File]::Exists($NsightCompute)) (
        "Nsight Compute does not exist: $NsightCompute"
    )

    $releases = @(
        [pscustomobject]@{
            Name = 'R2024b'
            Matlab = $MatlabR2024b
        },
        [pscustomobject]@{
            Name = 'R2025b'
            Matlab = $MatlabR2025b
        }
    )

    $runtimeRoot = Join-Path $script:RepositoryRoot 'build\f18-cuda-gate'
    New-F18Directory $runtimeRoot $script:RepositoryRoot
    $completedVersions = 0
    $completedProfiles = 0
    $validInvocations = 0
    $noKernelProfiles = 0

    foreach ($release in $releases) {
        Assert-F18 ([System.IO.File]::Exists($release.Matlab)) (
            "$($release.Name) executable does not exist: $($release.Matlab)"
        )

        $releaseRoot = Join-Path $runtimeRoot $release.Name
        $preferences = Join-Path $releaseRoot 'matlab-pref'
        $temporary = Join-Path $releaseRoot 'temp'
        foreach ($directory in @($releaseRoot, $preferences, $temporary)) {
            New-F18Directory $directory $script:RepositoryRoot
        }

        $profileResults = @{}
        foreach ($profile in @('valid', 'no-kernel')) {
            $batchPath = Join-Path $releaseRoot "f18_cuda_$profile.m"
            $logPath = Join-Path $releaseRoot "ncu-$profile.log"
            if ([System.IO.File]::Exists($logPath)) {
                Remove-Item -LiteralPath $logPath -Force
            }
            New-F18CudaBatch $profile $release.Name $batchPath

            $runExpression = 'run(' + (
                ConvertTo-MatlabLiteral $batchPath
            ) + ');'
            $arguments = @(
                '--set',
                'none',
                '--target-processes',
                'all',
                '--print-summary',
                'per-kernel',
                $release.Matlab,
                '-batch',
                $runExpression
            )
            Assert-NoProfilerCaps $arguments
            $environment = @{
                'MATLAB_PREFDIR' = $preferences
                'TEMP' = $temporary
                'TMP' = $temporary
            }
            $profileResults[$profile] = Invoke-CapturedNative (
                $NsightCompute
            ) $arguments $logPath $environment $script:RepositoryRoot
            Assert-F18 ($profileResults[$profile].ExitCode -eq 0) (
                "$($release.Name) $profile Nsight process exited " +
                "$($profileResults[$profile].ExitCode); see $logPath"
            )

            $pathMarker = (
                "F18_MEX_PATH version=$($release.Name) profile=$profile " +
                "path=$($script:CudaMexPath)"
            )
            Assert-F18 (
                (Get-LiteralCount $profileResults[$profile].Text $pathMarker) -eq 1
            ) (
                "$($release.Name) $profile profile did not resolve exactly " +
                "the registered CUDA MEX."
            )
            ++$completedProfiles
        }

        $validText = $profileResults['valid'].Text
        $validMarker = (
            "F18_CUDA_VALID version=$($release.Name) " +
            'l1_labels=[1 2 1 1] l1_medoids=[3 2] l1_cost=9 ' +
            'squared_labels=[2 1 1 1] squared_medoids=[4 1] ' +
            'squared_cost=30 active_l1=gpu explicit_squared=gpu ninit=2'
        )
        Assert-F18 ((Get-LiteralCount $validText $validMarker) -eq 1) (
            "$($release.Name) valid profile did not print its exact result marker."
        )
        Assert-F18 (
            $validText -notmatch '(?i)No kernels were profiled'
        ) "$($release.Name) valid profile launched no kernel."
        $validRows = @(Get-ProfilerInvocationRows $validText)
        Assert-F18 ($validRows.Count -gt 0) (
            "$($release.Name) valid profile has no parsed DTW summary row."
        )
        $releaseInvocations = 0
        foreach ($row in $validRows) {
            Assert-F18 ($row.Device -eq 0) (
                "$($release.Name) profiled Device $($row.Device), expected 0."
            )
            Assert-F18 ($row.ComputeCapability -eq '8.9') (
                "$($release.Name) profiled CC $($row.ComputeCapability), " +
                'expected 8.9.'
            )
            $releaseInvocations += $row.Invocations
        }
        Assert-F18 ($releaseInvocations -eq 2) (
            "$($release.Name) valid profile reported $releaseInvocations " +
            'DTW invocations, expected exactly 2.'
        )

        $noKernelText = $profileResults['no-kernel'].Text
        $noKernelMarker = (
            "F18_CUDA_NO_KERNEL version=$($release.Name) " +
            'cpu_labels=[1 2 1 1] cpu_medoids=[3 2] cpu_cost=9 ' +
            'cpu_override=1/1 variant_rejected=1/1 ' +
            'missing_rejected=1/1 device_unchanged=2/2'
        )
        Assert-F18 ((Get-LiteralCount $noKernelText $noKernelMarker) -eq 1) (
            "$($release.Name) no-kernel profile did not print its exact marker."
        )
        Assert-F18 (
            $noKernelText -match '(?i)No kernels were profiled'
        ) "$($release.Name) no-kernel control lacks Nsight's no-kernel record."
        $noKernelRows = @(Get-ProfilerInvocationRows $noKernelText)
        Assert-F18 ($noKernelRows.Count -eq 0) (
            "$($release.Name) no-kernel control contains a profiled invocation."
        )

        Write-Output (
            "F18_MATLAB_CUDA version=$($release.Name) " +
            'subject=DTWClustering.fit profiles=2/2 mex_paths=2/2 ' +
            'routes=3/3 gpu_results=2/2 gpu_kernels=2/2 ' +
            'invocations=2/2 overrides=2/2 gpu_rejections=2/2 ' +
            'no_kernel=1/1 skips=0'
        )
        ++$completedVersions
        $validInvocations += $releaseInvocations
        ++$noKernelProfiles
    }

    Assert-F18 ($completedVersions -eq 2) (
        "completed $completedVersions CUDA version gates, expected 2."
    )
    Assert-F18 ($completedProfiles -eq 4) (
        "completed $completedProfiles CUDA profiles, expected 4."
    )
    Assert-F18 ($validInvocations -eq 4) (
        "observed $validInvocations valid invocations, expected 4."
    )
    Assert-F18 ($noKernelProfiles -eq 2) (
        "observed $noKernelProfiles no-kernel profiles, expected 2."
    )
    Write-Output (
        'F18_MATLAB_CUDA subject=DTWClustering.fit versions=2/2 ' +
        'profiles=4/4 valid_invocations=4/4 no_kernel_profiles=2/2 ' +
        'skips=0'
    )
}

switch ($Mode) {
    'source' {
        Test-F18Source
    }
    'hpc' {
        Test-F18Hpc
    }
    'cuda' {
        Test-F18Cuda
    }
    'all' {
        Test-F18Source
        Test-F18Hpc
        Test-F18Cuda
    }
}
