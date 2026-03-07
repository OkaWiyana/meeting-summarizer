# run.ps1 — Jalankan Meeting Summarizer dengan satu klik
# Usage: klik kanan → "Run with PowerShell"  ATAU  ketik: .\run.ps1

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Activate = Join-Path $Root "venv\Scripts\Activate.ps1"

if (Test-Path $Activate) {
    & $Activate
    Write-Host "Virtual environment aktif." -ForegroundColor Green
} else {
    Write-Host ".venv tidak ditemukan. Pastikan kamu sudah membuat virtual environment." -ForegroundColor Yellow
}

Write-Host "Menjalankan Streamlit..." -ForegroundColor Cyan
streamlit run "$Root\app.py"
