@echo off
setlocal

set "ROOT=%~dp0"
set "CMD=%ROOT%.venv\Scripts\chemverify.exe"
set "LOCAL_NODE=%ROOT%.local\node\current"

if exist "%LOCAL_NODE%\node.exe" (
  set "PATH=%LOCAL_NODE%;%PATH%"
)

if not exist "%CMD%" (
  echo ChemVerify is not installed yet. Run:
  echo powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1
  exit /b 1
)

"%CMD%" %*
