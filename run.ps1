param(
    [string]$Reference,
    [string]$Video,
    [int[]]$ReferenceBox,
    [string]$SceneContext = '',
    [string]$Model = 'qwen3.5:0.8b',
    [double]$Threshold = 6.0,
    [double]$VerifyEvery = 10,
    [switch]$SetupOnly
)
$ErrorActionPreference = 'Stop'
$taskPython = $null
foreach ($taskCandidate in @('py', 'python')) {
    if (Get-Command $taskCandidate -ErrorAction SilentlyContinue) {
        $taskPrefix = @()
        if ($taskCandidate -eq 'py') { $taskPrefix = @('-3.11') }
        $taskResolved = & $taskCandidate @taskPrefix -c 'import sys; print(sys.executable) if sys.version_info >= (3,11) else sys.exit(1)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $taskResolved) { $taskPython = $taskResolved.Trim(); break }
    }
}
if (-not $taskPython) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'Install Python 3.11 from python.org, then repeat this command.'
    }
    Write-Output 'Installing Python 3.11 for the current user...'
    & winget install --exact --id Python.Python.3.11 --scope user --silent --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
    $taskPython = Join-Path $env:LOCALAPPDATA 'Programs/Python/Python311/python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Restart the terminal and repeat this command.' }
}
$taskArguments = @((Join-Path $PSScriptRoot 'run.py'), '--model', $Model,
    '--tau', $Threshold.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--verify-every', $VerifyEvery.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($Reference) { $taskArguments += @('--reference', $Reference) }
if ($Video) { $taskArguments += @('--video', $Video) }
if ($SceneContext) { $taskArguments += @('--scene-context', $SceneContext) }
if ($ReferenceBox.Count -gt 0) {
    if ($ReferenceBox.Count -ne 4) { throw 'ReferenceBox requires four coordinates.' }
    $taskArguments += '--ref-box'
    $taskArguments += @($ReferenceBox | ForEach-Object { [string]$_ })
}
if ($SetupOnly) { $taskArguments += '--setup-only' }
& $taskPython @taskArguments
if ($LASTEXITCODE -ne 0) { throw 'Setup or video processing failed. See the error above.' }
