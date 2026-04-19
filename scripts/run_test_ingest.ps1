<#
.SYNOPSIS
    Run the MTSS ingest pipeline against a sandboxed sample of EMLs.

.DESCRIPTION
    Generic integration harness. Runs `mtss ingest` and `mtss validate ingest`
    against `data/test_ingest/emails/` -> `data/test_ingest/output/`, completely
    isolated from the production `data/output/` state.

    Drop any number of `.eml` files into `data/test_ingest/emails/` and re-run
    this script. The harness is content-agnostic: it does not care how many
    EMLs are present or which attachments they carry.

.PARAMETER NoValidate
    Skip the post-ingest `mtss validate ingest` step.

.PARAMETER Lenient
    Pass `--lenient` to ingest (continue past per-file errors).

.PARAMETER Reset
    Wipe `data/test_ingest/output/` before ingesting (forces a from-scratch
    run; otherwise progress resumes via processing_log.jsonl).

.PARAMETER Limit
    Cap the number of EMLs processed this run (forwarded to `mtss ingest -n`).

.EXAMPLE
    pwsh ./scripts/run_test_ingest.ps1
    pwsh ./scripts/run_test_ingest.ps1 -Reset -Lenient -Limit 5

.NOTES
    NEVER touches `data/output/`. Production state is read-only from this
    script's perspective.
#>

[CmdletBinding()]
param(
    [switch]$NoValidate,
    [switch]$Lenient,
    [switch]$Reset,
    [int]$Limit = 0
)

$ErrorActionPreference = 'Stop'

# Resolve repo root from this script's location (works regardless of cwd).
$RepoRoot   = Resolve-Path (Join-Path $PSScriptRoot '..')
$SourceDir  = Join-Path $RepoRoot 'data\test_ingest\emails'
$OutputDir  = Join-Path $RepoRoot 'data\test_ingest\output'

# --- Sanity: ensure we will not touch production data/output --------------
$ProdOutput = (Join-Path $RepoRoot 'data\output').ToLower()
if ($OutputDir.ToLower() -eq $ProdOutput) {
    Write-Error "Refusing to run: resolved output dir collides with production data/output."
    exit 1
}

if (-not (Test-Path $SourceDir)) {
    Write-Error "Source dir not found: $SourceDir"
    Write-Host  "Create it and drop sample .eml files inside, then re-run." -ForegroundColor Yellow
    exit 1
}

$emlFiles = Get-ChildItem -Path $SourceDir -Filter '*.eml' -File -ErrorAction SilentlyContinue
if ($emlFiles.Count -eq 0) {
    Write-Error "No .eml files in $SourceDir."
    exit 1
}

Write-Host ""
Write-Host "=== MTSS test-ingest harness ===" -ForegroundColor Cyan
Write-Host ("Source:  {0}" -f $SourceDir)
Write-Host ("Output:  {0}" -f $OutputDir)
Write-Host ("EMLs:    {0}" -f $emlFiles.Count)
Write-Host ""

if ($Reset -and (Test-Path $OutputDir)) {
    Write-Host "[reset] Removing $OutputDir ..." -ForegroundColor Yellow
    Remove-Item -Path $OutputDir -Recurse -Force
}

# --- Build the ingest command --------------------------------------------
# Confirmed CLI surface (src/mtss/cli/ingest_cmd.py):
#   --source / -s, --output-dir / -o, --lenient, --limit / -n
$ingestArgs = @(
    'run', 'mtss', 'ingest',
    '--source',     $SourceDir,
    '--output-dir', $OutputDir
)
if ($Lenient)    { $ingestArgs += '--lenient' }
if ($Limit -gt 0) { $ingestArgs += @('--limit', $Limit) }

Write-Host "[ingest] uv $($ingestArgs -join ' ')" -ForegroundColor Cyan
& uv @ingestArgs
$ingestExit = $LASTEXITCODE
if ($ingestExit -ne 0) {
    Write-Host "[ingest] uv exit code: $ingestExit" -ForegroundColor Yellow
}

# --- Validate ------------------------------------------------------------
$validateExit = 0
if (-not $NoValidate) {
    Write-Host ""
    Write-Host "[validate] uv run mtss validate ingest --output-dir $OutputDir" -ForegroundColor Cyan
    & uv run mtss validate ingest --output-dir $OutputDir
    $validateExit = $LASTEXITCODE
}

# --- Summary -------------------------------------------------------------
Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan

function Read-JsonLines([string]$Path) {
    if (-not (Test-Path $Path)) { return @() }
    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($line in Get-Content -LiteralPath $Path -Encoding utf8) {
        $t = $line.Trim()
        if ($t.Length -eq 0) { continue }
        try { $rows.Add(($t | ConvertFrom-Json)) } catch { }
    }
    return ,$rows
}

$procLog  = Read-JsonLines (Join-Path $OutputDir 'processing_log.jsonl')
$docs     = Read-JsonLines (Join-Path $OutputDir 'documents.jsonl')
$chunks   = Read-JsonLines (Join-Path $OutputDir 'chunks.jsonl')
$topics   = Read-JsonLines (Join-Path $OutputDir 'topics.jsonl')
$events   = Read-JsonLines (Join-Path $OutputDir 'ingest_events.jsonl')

# Latest-status-per-file (processing_log.jsonl appends)
$latestByFile = @{}
foreach ($p in $procLog) { $latestByFile[$p.file_path] = $p }
$completed = @($latestByFile.Values | Where-Object { $_.status -eq 'COMPLETED' }).Count
$failed    = @($latestByFile.Values | Where-Object { $_.status -eq 'FAILED' }).Count
$other     = $latestByFile.Count - $completed - $failed

Write-Host ("Files (processing_log): {0} completed / {1} failed / {2} other" -f $completed, $failed, $other)
Write-Host ("Documents: {0}" -f $docs.Count)
Write-Host ("Chunks:    {0}" -f $chunks.Count)
Write-Host ("Topics:    {0}" -f $topics.Count)
Write-Host ("Events:    {0}" -f $events.Count)

if ($docs.Count -gt 0) {
    $modeBuckets = @{}
    foreach ($d in $docs) {
        $m = if ($d.embedding_mode) { $d.embedding_mode } else { '(unset)' }
        if ($modeBuckets.ContainsKey($m)) { $modeBuckets[$m]++ } else { $modeBuckets[$m] = 1 }
    }
    Write-Host ""
    Write-Host "Embedding-mode distribution (documents):"
    foreach ($k in ($modeBuckets.Keys | Sort-Object)) {
        Write-Host ("  {0,-15} {1}" -f $k, $modeBuckets[$k])
    }
}

if ($failed -gt 0) {
    Write-Host ""
    Write-Host "Failed files:" -ForegroundColor Red
    foreach ($p in ($latestByFile.Values | Where-Object { $_.status -eq 'FAILED' })) {
        $err = if ($p.error) { $p.error } else { '(no error message)' }
        Write-Host ("  {0}  ->  {1}" -f $p.file_path, $err)
    }
}

Write-Host ""
if ($ingestExit -ne 0) {
    Write-Host "ingest exited with code $ingestExit" -ForegroundColor Yellow
}
if ($validateExit -ne 0) {
    Write-Host "validate exited with code $validateExit (issues found)" -ForegroundColor Yellow
}
exit ([Math]::Max($ingestExit, $validateExit))
