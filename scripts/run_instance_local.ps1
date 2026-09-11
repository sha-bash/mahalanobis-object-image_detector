param(
    [string]$Reference = 'data/refs/frame_000000.jpg',
    [string]$Video = 'data/search/video_test.mp4',
    [int[]]$ReferenceBox = @(41, 34, 229, 207),
    [string]$Model = 'qwen3.5:0.8b',
    [double]$Threshold = 6.0,
    [double]$VerifyEvery = 10,
    [string]$SceneContext = 'Outdoor aerial drone footage of objects on the ground',
    [switch]$SetupOnly
)
# Compatibility entry for the supplied experiment; setup lives only in run.py.
$taskParameters = @{
    Reference = $Reference; Video = $Video; ReferenceBox = $ReferenceBox
    Model = $Model; Threshold = $Threshold; VerifyEvery = $VerifyEvery
    SceneContext = $SceneContext; SetupOnly = $SetupOnly
}
& (Join-Path (Split-Path $PSScriptRoot -Parent) 'run.ps1') @taskParameters
