$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath '.python-runtime\python.exe') {
    $projectPython = Join-Path $PSScriptRoot '.python-runtime\python.exe'
} else {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or 3.12 is required.' }
    }
    $projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
& $projectPython -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $projectPython train.py --database data/european_players.sqlite --output artifacts/european_model.joblib
if ($LASTEXITCODE -ne 0) { throw 'Valuation model training failed.' }
& $projectPython prepare_forecast.py
if ($LASTEXITCODE -ne 0) { throw 'Forecast preparation failed.' }
& $projectPython train.py --database data/forecast_players.sqlite --task next-season --output artifacts/forecast_model.joblib
if ($LASTEXITCODE -ne 0) { throw 'Forecast training failed.' }
Write-Output 'Setup complete. Run ./run.ps1 to start the dashboard.'
