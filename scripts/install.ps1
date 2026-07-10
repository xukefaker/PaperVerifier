$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

Write-Host "ChemSearch installer" -ForegroundColor Cyan

$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $UvCommand) {
  $UvCandidates = @(
    (Join-Path $HOME ".local\bin\uv.exe"),
    (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe")
  )
  $UvPath = $UvCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
} else {
  $UvPath = $UvCommand.Source
}

if (-not $UvPath) {
  Write-Host "uv is required. Install it first:" -ForegroundColor Red
  Write-Host "winget install --id=astral-sh.uv -e"
  exit 1
}

function Test-NodeOk {
  $node = Get-Command node -ErrorAction SilentlyContinue
  if (-not $node) {
    return $false
  }
  node -e "process.exit(Number(process.versions.node.split('.')[0]) >= 20 ? 0 : 1)" *> $null
  return ($LASTEXITCODE -eq 0)
}

function Install-LocalNode {
  $NodeVersion = if ($env:CHEMSEARCH_NODE_VERSION) { $env:CHEMSEARCH_NODE_VERSION } else { "22.13.1" }
  $Arch = switch ($env:PROCESSOR_ARCHITECTURE) {
    "AMD64" { "x64" }
    "ARM64" { "arm64" }
    default {
      Write-Host "Unsupported CPU for automatic Node.js install: $env:PROCESSOR_ARCHITECTURE" -ForegroundColor Red
      exit 1
    }
  }

  $NodeRoot = Join-Path $Root ".local\node"
  $Current = Join-Path $NodeRoot "current"
  $NodeExe = Join-Path $Current "node.exe"
  if (Test-Path $NodeExe) {
    & $NodeExe -e "process.exit(Number(process.versions.node.split('.')[0]) >= 20 ? 0 : 1)" *> $null
    if ($LASTEXITCODE -eq 0) {
      $env:PATH = "$Current;$env:PATH"
      Write-Host "Using local Node.js $(& $NodeExe -v)"
      return
    }
  }

  $Name = "node-v$NodeVersion-win-$Arch"
  $ZipName = "$Name.zip"
  $Url = "https://nodejs.org/dist/v$NodeVersion/$ZipName"
  $ZipPath = Join-Path $NodeRoot $ZipName
  $ExtractPath = Join-Path $NodeRoot "extract"
  New-Item -ItemType Directory -Force -Path $NodeRoot | Out-Null
  Write-Host "Node.js >=20 not found. Installing local Node.js $NodeVersion..."
  Invoke-WebRequest -Uri $Url -OutFile $ZipPath
  Remove-Item $ExtractPath -Recurse -Force -ErrorAction SilentlyContinue
  Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force
  Remove-Item $Current -Recurse -Force -ErrorAction SilentlyContinue
  Move-Item -Path (Join-Path $ExtractPath $Name) -Destination $Current
  Remove-Item $ExtractPath -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
  $env:PATH = "$Current;$env:PATH"
}

if (Test-NodeOk) {
  Write-Host "Using system Node.js $(node -v)"
} else {
  Install-LocalNode
}

& $UvPath python install 3.12
& $UvPath venv --python 3.12 --allow-existing .venv

$env:VIRTUAL_ENV = Join-Path $Root ".venv"
$env:PATH = (Join-Path $env:VIRTUAL_ENV "Scripts") + ";" + $env:PATH

& $UvPath pip install -e . --torch-backend=auto
& "$Root\chemsearch.cmd" init
& "$Root\chemsearch.cmd" doctor

Write-Host ""
Write-Host "Done. Edit .env, then run:" -ForegroundColor Green
Write-Host ".\chemsearch.cmd demo-chem --max-papers 5"
Write-Host ".\chemsearch.cmd index"
Write-Host ".\chemsearch.cmd web"
