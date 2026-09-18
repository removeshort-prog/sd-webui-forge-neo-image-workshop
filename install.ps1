param([string]$ForgePath)

$ErrorActionPreference = 'Stop'
try {
    if (-not $ForgePath) {
        Write-Host 'Paste the Forge ROOT folder (the folder containing launch.py).'
        $ForgePath = Read-Host 'Forge folder'
    }
    $ForgePath = $ForgePath.Trim().Trim('"')
    if (-not $ForgePath) { throw 'No folder was provided.' }
    $resolvedRoot = (Resolve-Path -LiteralPath $ForgePath).Path
    if (-not (Test-Path -LiteralPath (Join-Path $resolvedRoot 'launch.py') -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $resolvedRoot 'modules') -PathType Container)) {
        throw 'This does not look like a Forge root folder.'
    }
    $extensionDir = Join-Path $resolvedRoot 'extensions'
    $destination = Join-Path $extensionDir 'sd-webui-forge-neo-image-workshop'
    $legacyDestination = Join-Path $extensionDir 'sd-webui-forge-image-workshop'
    if (Test-Path -LiteralPath $legacyDestination) {
        throw "An earlier plugin folder exists: $legacyDestination. Move it outside extensions first."
    }
    if (Test-Path -LiteralPath $destination) {
        throw "Plugin folder already exists: $destination. Move the old version outside extensions first."
    }
    New-Item -ItemType Directory -Path $extensionDir -Force | Out-Null
    Copy-Item -LiteralPath $PSScriptRoot -Destination $destination -Recurse
    Write-Host "Installed: $destination" -ForegroundColor Green
    Write-Host 'Fully restart Forge, then open the Image Workshop tab.'
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
