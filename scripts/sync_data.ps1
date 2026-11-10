<#
.SYNOPSIS
    Download all of the paper's data from S3 into one local folder.

.DESCRIPTION
    notebooks/analyses.py reads only this local copy (ZEPHYR_DATA, default
    D:\zephyr-data), never S3. This script makes it with `aws s3 sync`, laid out as the
    Code Ocean capsule mounts the data (/data/...), plus the runs:

        <Root>/zephyr-dataset/           the corpus (videos, thermistors, clips.toml, raw/)
        <Root>/zephyr-df-dataset/        the dynamic-foraging sessions
        <Root>/zephyr-cache/<recipe>/    the feature caches the runs trained and scored on
                                         (zephyr-cache-v2 on S3)
        <Root>/zephyr-manuscript-runs/   every run (and analysis) the paper reports
        <Root>/aind-open-data/           of each session without a thermistor, its face
                                         camera (video and clock); the processed assets
                                         the figures use

    About 2.1 TB plus the aind-open-data sessions. Run it again at any time: `sync`
    copies only what is new or changed. It writes sync.json (what was copied, from
    where, when), which analyses.py records in every provenance.json.

    Needs the AWS CLI (v2) and credentials that can read the bucket (`aws sso login`).

.PARAMETER Root
    The local folder. Default D:\zephyr-data.

.PARAMETER Bucket
    The paper's bucket prefix. Default: ZEPHYR_BUCKET, else [tool.zephyr-manuscript]
    bucket in pyproject.toml.

.PARAMETER DryRun
    List what would be copied, copy nothing.

.EXAMPLE
    .\scripts\sync_data.ps1 -DryRun
    .\scripts\sync_data.ps1
#>
param(
    [string]$Root = "D:\zephyr-data",
    [string]$Bucket = "",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"

if (-not $Bucket) {
    if ($env:ZEPHYR_BUCKET) {
        $Bucket = $env:ZEPHYR_BUCKET
    } else {
        $pyproject = Join-Path $PSScriptRoot "..\pyproject.toml"
        $line = Select-String -Path $pyproject -Pattern '^bucket\s*=\s*"([^"]+)"' | Select-Object -First 1
        if (-not $line) { throw "No bucket in ${pyproject}: pass -Bucket or set ZEPHYR_BUCKET." }
        $Bucket = $line.Matches[0].Groups[1].Value
    }
}
$Bucket = $Bucket.TrimEnd("/")
if (-not $Bucket.StartsWith("s3://")) { $Bucket = "s3://$Bucket" }

aws sts get-caller-identity --output text | Out-Null
if ($LASTEXITCODE -ne 0) { throw "No AWS credentials: run 'aws sso login' first." }

# The sessions without a thermistor (clips-application.toml) and the processed assets
# the figures read (the application's, and the Fig. 1b-c session's): on aind-open-data.
$configs = Join-Path $PSScriptRoot "..\packages\zephyr-benchmarks\configs"
$sessions = Select-String -Path (Join-Path $configs "manuscript\clips\clips-application.toml") `
    -Pattern '^group\s*=\s*"(.+)_FaceCamera"' | ForEach-Object { $_.Matches[0].Groups[1].Value }
$assets = Select-String -Path (Join-Path $configs "results\*.toml") `
    -Pattern '^asset\s*=\s*"([^"]+)"' | ForEach-Object { $_.Matches[0].Groups[1].Value }
$assets += Select-String -Path (Join-Path $PSScriptRoot "..\notebooks\manuscript_figures.py") `
    -Pattern '"(\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_processed_[0-9_-]+)"' -AllMatches |
    ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value }
$sessions = $sessions | Sort-Object -Unique
$assets = $assets | Sort-Object -Unique

# S3 source -> local folder under Root.
$parts = @(
    @{ From = "$Bucket/zephyr-manuscript-runs"; To = "zephyr-manuscript-runs" },
    @{ From = "$Bucket/zephyr-dataset";         To = "zephyr-dataset" },
    @{ From = "$Bucket/zephyr-df-dataset";      To = "zephyr-df-dataset" },
    @{ From = "$Bucket/zephyr-cache-v2";        To = "zephyr-cache" }
)
# Of a raw session (about 26 GB, every camera), only the face camera's video and clock.
$parts += $sessions | ForEach-Object { @{
    From = "s3://aind-open-data/$_/behavior-videos/FaceCamera"; To = "aind-open-data\$_\behavior-videos\FaceCamera" } }
$parts += $assets | ForEach-Object { @{ From = "s3://aind-open-data/$_"; To = "aind-open-data\$_" } }

New-Item -ItemType Directory -Force -Path $Root | Out-Null
$log = Join-Path $Root "sync.log"
foreach ($p in $parts) {
    $target = Join-Path $Root $p.To
    $cmd = @("s3", "sync", $p.From, $target, "--no-progress")
    $cmd += $(if ($DryRun) { "--dryrun" } else { "--only-show-errors" })
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  aws $($cmd -join ' ')" | Tee-Object -FilePath $log -Append
    & aws @cmd
    if ($LASTEXITCODE -ne 0) { throw "aws s3 sync $($p.From) failed (exit $LASTEXITCODE): run the script again to resume." }
}

if (-not $DryRun) {
    [ordered]@{
        bucket       = $Bucket
        root         = (Resolve-Path $Root).Path
        finished_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        parts        = $parts | ForEach-Object { [ordered]@{ from = $_.From; to = $_.To } }
    } | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $Root "sync.json") -Encoding utf8
}
Write-Host "Done: $Root (log $log)."
