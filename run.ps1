$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
    & '.\.venv\Scripts\python.exe' -m streamlit run app.py
} elseif (Test-Path -LiteralPath '.python-runtime\python.exe') {
    & '.\.python-runtime\python.exe' -m streamlit run app.py
} else {
    Write-Error 'Install Python and follow the setup steps in README.md first.'
}
