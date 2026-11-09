<#
.SYNOPSIS
    Run notebooks/analyses.py headless, from a clean commit, into one output folder.

.DESCRIPTION
    Runs every analysis the current commit has not made yet (or those -Only names, but
    those -Skip names) as a script, reading the local copy of the paper's data (-Data,
    from scripts/sync_data.ps1) and writing one folder per analysis into -Out, each with
    its provenance.json. Upload them afterwards with
    scripts/upload_analyses.ps1 -Source <Out>. The output is also logged to
    <Out>\analyses-<time>.log.

    Refuses to start if the working tree differs from the commit (tracked or untracked
    files), as every analysis would.

.PARAMETER Out
    Where the analyses are written (ZEPHYR_ANALYSES). Default notebooks/cache/analyses.

.PARAMETER Data
    The local copy of the data and runs (ZEPHYR_DATA). Default D:\zephyr-data.

.PARAMETER Only
    Run only these analyses (e.g. checks,tables).

.PARAMETER Skip
    Run all but these analyses (e.g. timing).

.PARAMETER Force
    Make again the analyses this commit already made.

.EXAMPLE
    .\scripts\run_analyses.ps1 -Out D:\data\zephyr-temp-output
    .\scripts\run_analyses.ps1 -Out D:\data\zephyr-temp-output -Skip timing
    .\scripts\run_analyses.ps1 -Out D:\data\zephyr-temp-output -Only tables -Force
#>
param(
    [string]$Out = (Join-Path $PSScriptRoot "..\notebooks\cache\analyses"),
    [string]$Data = "D:\zephyr-data",
    [string[]]$Only = @(),
    [string[]]$Skip = @(),
    [switch]$Force
)
$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repo

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv is not on PATH." }
}
if (-not (Test-Path $Data -PathType Container)) {
    throw "No data copy at ${Data}: pass -Data (or run scripts/sync_data.ps1 first)."
}
$changed = git status --porcelain
if ($changed) { throw "Commit first: the analyses run only from a clean working tree.`n$($changed -join "`n")" }

New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$env:ZEPHYR_DATA = $Data
$env:ZEPHYR_ANALYSES = $Out
$env:PYTHONUNBUFFERED = "1"
# Another project's activated venv would only make uv warn; the run uses this repo's.
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue

$cmd = @("run", "--locked", "--all-packages", "--extra", "gpu", "python", "notebooks/analyses.py")
$extra = @()
if ($Only) { $extra += @("--only", ($Only -join ",")) }
if ($Skip) { $extra += @("--skip", ($Skip -join ",")) }
if ($Force) { $extra += "--force" }
if ($extra) { $cmd += @("--") + $extra }

$log = Join-Path $Out ("analyses-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))
Write-Host "commit $(git rev-parse --short HEAD)  data $Data  ->  $Out"
Write-Host "uv $($cmd -join ' ')"
Write-Host "log $log"

# Windows PowerShell turns each stderr line (warnings, progress) into an error record,
# which "Stop" would make fatal: judge the run by its exit code instead.
$ErrorActionPreference = "Continue"
# The log in UTF-8 (Windows PowerShell's Tee-Object writes UTF-16), flushed per line.
$writer = [System.IO.StreamWriter]::new($log, $false, [System.Text.UTF8Encoding]::new($false))
try {
    & uv @cmd 2>&1 | ForEach-Object { $line = "$_"; $writer.WriteLine($line); $writer.Flush(); $line }
    $code = $LASTEXITCODE
} finally {
    $writer.Close()
}
$ErrorActionPreference = "Stop"
if ($code -ne 0) { throw "analyses.py failed (exit $code); see $log." }
Write-Host "Done. Upload with: .\scripts\upload_analyses.ps1 -Source $Out -DryRun"
