<#
Recreate the tested Windows x64 / Python 3.10 / CUDA 12.6 environment.
Usage from the project directory:
  & .\scripts\setup_environment.ps1 -PythonExe 'C:\path\to\python.exe'
Use a Python 3.10 installation (validated with 3.10.21). The base installation
must remain available on this computer. On another computer, rebuild .venv;
do not copy it. The script does not download datasets or model checkpoints.
#>
[CmdletBinding()]
param([string]$PythonExe = 'python')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$lockFile = Join-Path $projectRoot 'requirements.lock.txt'

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $lockFile)) {
    throw 'requirements.lock.txt is required; use the validated project version.'
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    $venvRoot = Join-Path $projectRoot '.venv'
    if (Test-Path -LiteralPath $venvRoot) {
        throw '.venv exists but has no Python executable. Inspect it before recreating.'
    }
    Invoke-CheckedPython -Executable $PythonExe -Arguments @(
        '-c', 'import sys,struct; assert sys.platform == "win32" and struct.calcsize("P") == 8; assert sys.version_info[:2] == (3,10), "Select Python 3.10.x with -PythonExe"'
    )
    Invoke-CheckedPython -Executable $PythonExe -Arguments @('-m', 'venv', $venvRoot)
}
Invoke-CheckedPython -Executable $venvPython -Arguments @(
    '-c', 'import sys,struct; assert sys.platform == "win32" and struct.calcsize("P") == 8; assert sys.version_info[:2] == (3,10); assert sys.prefix != sys.base_prefix'
)
Invoke-CheckedPython -Executable $venvPython -Arguments @(
    '-m', 'pip', 'install', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', 'pip==26.2.2'
)
# Obtain only the CUDA wheels from PyTorch. All other dependencies are installed
# from PyPI using the complete lock below. This avoids mixing package indexes.
Invoke-CheckedPython -Executable $venvPython -Arguments @(
    '-m', 'pip', 'install', '--only-binary=:all:', '--no-deps',
    '--index-url', 'https://download.pytorch.org/whl/cu126',
    'torch==2.10.0+cu126', 'torchvision==0.25.0+cu126'
)
Invoke-CheckedPython -Executable $venvPython -Arguments @(
    '-m', 'pip', 'install', '--only-binary=:all:',
    '--index-url', 'https://pypi.org/simple', '-r', $lockFile
)
Invoke-CheckedPython -Executable $venvPython -Arguments @('-m', 'pip', 'check')
Invoke-CheckedPython -Executable $venvPython -Arguments @((Join-Path $PSScriptRoot 'check_environment.py'))
Write-Host "Environment ready: $venvPython"
