param(
    [ValidateRange(1024, 65535)][int]$Port = 8503,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repo
try {
    & $Python -m streamlit run demo/app.py --server.address 127.0.0.1 --server.port $Port --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false --client.toolbarMode minimal --theme.base light
    if ($LASTEXITCODE -ne 0) { throw "Demo stopped with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
