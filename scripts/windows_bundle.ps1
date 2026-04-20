$ErrorActionPreference = "Stop"

$ARCH = if ($env:ARCH) { $env:ARCH } else { "x86_64" }
$PLATFORM = "windows-${ARCH}"

pyinstaller installer\CGAspects.py --onedir --windowed --name=CGAspects `
    --icon=res\app_icons\CGAspects.ico `
    --noconfirm --exclude-module=pytest 

Write-Host "Build completed successfully."
