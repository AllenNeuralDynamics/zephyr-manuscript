<#
.SYNOPSIS
    Upload the analyses notebooks/analyses.py made into the paper's runs on S3.

.DESCRIPTION
    Copies, with `aws s3 sync`, each analysis folder of -Source (one folder per
    analysis, as the notebook writes them) to <bucket>/zephyr-manuscript-runs/analysis/,
    where notebooks/manuscript_figures.py reads them. Only folders whose provenance.json
    says they were made from a clean, committed working tree are uploaded; any other is
    refused and named. Nothing is deleted on S3.

    Needs the AWS CLI (v2) and credentials that can write the bucket (`aws sso login`).

.PARAMETER Source
    The notebook's output folder. Default notebooks/cache/analyses.

.PARAMETER Bucket
    The paper's bucket prefix. Default: ZEPHYR_BUCKET, else [tool.zephyr-manuscript]
    bucket in pyproject.toml.

.PARAMETER Only
    Upload only these analyses (folder names); default every one.

.PARAMETER DryRun
    List what would be copied, copy nothing.

.EXAMPLE
    .\scripts\upload_analyses.ps1 -DryRun
    .\scripts\upload_analyses.ps1
    .\scripts\upload_analyses.ps1 -Only noise-null,timing
#>
param(
    [string]$Source = (Join-Path $PSScriptRoot "..\notebooks\cache\analyses"),
    [string]$Bucket = "",
    [string[]]$Only = @(),
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"

if (-not $Bucket) {
    if ($env:ZEPHYR_BUCKET) {
        $Bucket = $env:ZEPHYR_BUCKET
    } else {
        $pyproject = Join-Path $PSScriptRoot "..\pyproject.toml"
        $line = Select-String -Path $pyproject -Pattern '^bucket\s*=\s*"([^"]+)"' | Select-Object -First 1
        if (-not $line) { throw "No bucket in $pyproject; pass -Bucket or set ZEPHYR_BUCKET." }
        $Bucket = $line.Matches[0].Groups[1].Value
    }
}
$Bucket = $Bucket.TrimEnd("/")
if (-not $Bucket.StartsWith("s3://")) { $Bucket = "s3://$Bucket" }
$Dest = "$Bucket/zephyr-manuscript-runs/analysis"

if (-not (Test-Path $Source)) { throw "No analyses at ${Source}: run notebooks/analyses.py first." }
$folders = Get-ChildItem -Path $Source -Directory
if ($Only) { $folders = $folders | Where-Object { $Only -contains $_.Name } }
if (-not $folders) { throw "Nothing to upload in $Source." }

# Every analysis must say it was made from a clean commit.
$refused = @()
foreach ($f in $folders) {
    $prov = Join-Path $f.FullName "provenance.json"
    if (-not (Test-Path $prov)) { $refused += "$($f.Name): no provenance.json"; continue }
    $p = Get-Content $prov -Raw | ConvertFrom-Json
    if ($p.dirty -ne $false -or -not $p.commit) { $refused += "$($f.Name): not made from a clean commit"; continue }
}
if ($refused) { throw "Refused, nothing uploaded:`n  $($refused -join "`n  ")" }

aws sts get-caller-identity --output text | Out-Null
if ($LASTEXITCODE -ne 0) { throw "No AWS credentials: run 'aws sso login' first." }

foreach ($f in $folders) {
    $p = Get-Content (Join-Path $f.FullName "provenance.json") -Raw | ConvertFrom-Json
    $cmd = @("s3", "sync", $f.FullName, "$Dest/$($f.Name)", "--no-progress")
    if ($DryRun) { $cmd += "--dryrun" }
    Write-Host "$($f.Name)  (commit $($p.commit))  ->  $Dest/$($f.Name)"
    & aws @cmd
    if ($LASTEXITCODE -ne 0) { throw "aws s3 sync of $($f.Name) failed (exit $LASTEXITCODE)." }
}
Write-Host "Done."
