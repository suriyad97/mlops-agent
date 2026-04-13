# ─────────────────────────────────────────────────────────────────────────────
# MLOps Agent — Windows PowerShell Launcher
# Usage: .\run.ps1
# ─────────────────────────────────────────────────────────────────────────────

Write-Host "⚡ Starting MLOps Agent..." -ForegroundColor Cyan

# Load .env
if (!(Test-Path ".env")) {
    Write-Host "✗ .env not found. Copy .env.example to .env and fill in your values." -ForegroundColor Red
    exit 1
}

# Install package in editable mode if not already installed
$installed = pip show mlops-agent 2>$null
if (!$installed) {
    Write-Host "📦 Installing mlops-agent package..." -ForegroundColor Yellow
    pip install -e ".[dev]" --quiet
}

# Start FastAPI backend on port 8001
Write-Host "▶ Starting FastAPI backend on http://localhost:8001 ..." -ForegroundColor Green
$backend = Start-Process -PassThru -NoNewWindow -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "backend.main:app", "--port", "8001", "--reload"

Start-Sleep -Seconds 2

# Start Chainlit frontend on port 8000
Write-Host "▶ Starting Chainlit UI on http://localhost:8000 ..." -ForegroundColor Green
try {
    python -m chainlit run frontend/chainlit_app.py --port 8000 --host 0.0.0.0
} finally {
    Write-Host "✓ Stopping backend..." -ForegroundColor Yellow
    Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    Write-Host "✓ Done." -ForegroundColor Green
}
