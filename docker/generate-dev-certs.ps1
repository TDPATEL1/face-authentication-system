# Self-signed certificate for local HTTPS only. Do not use in production.
# Do not commit server.key.

$ErrorActionPreference = "Stop"

$certDir = Join-Path $PSScriptRoot "..\nginx\certs"
New-Item -ItemType Directory -Force -Path $certDir | Out-Null

$openssl = $null
$candidates = @(
    "openssl",
    (Join-Path ${env:ProgramFiles} "Git\usr\bin\openssl.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Git\usr\bin\openssl.exe")
)

foreach ($candidate in $candidates) {
    if ($candidate -eq "openssl") {
        $cmd = Get-Command openssl -ErrorAction SilentlyContinue
        if ($cmd) {
            $openssl = $cmd.Source
            break
        }
    }
    elseif (Test-Path $candidate) {
        $openssl = $candidate
        break
    }
}

if (-not $openssl) {
    throw "openssl not found. Install Git for Windows or OpenSSL, then re-run this script."
}

$keyPath = Join-Path $certDir "server.key"
$crtPath = Join-Path $certDir "server.crt"

& $openssl req -x509 -nodes -newkey rsa:2048 `
    -keyout $keyPath `
    -out $crtPath `
    -days 365 `
    -subj "/CN=localhost"

if ($LASTEXITCODE -ne 0) {
    throw "openssl failed with exit code $LASTEXITCODE"
}

Write-Host "Wrote $crtPath and $keyPath"
