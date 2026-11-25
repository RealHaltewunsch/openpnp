# Windows .exe build — run on the Windows host from the repo (scripts/build-windows.ps1).
# Repo path is resolved relative to this script; tool discovery uses PATH/env vars
# and the current user's profile. No personal paths are hardcoded.
# Prereq (one-time setup, see docs/DINOv3-and-Packaging.md §4.2): Temurin JDK 21,
# Maven 3.x, WiX 3.x.
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot

# --- tool discovery (env vars win; otherwise discover under the user profile) ---
if (-not $env:JAVA_HOME) {
  $jdk = (Get-ChildItem 'C:\Program Files\Eclipse Adoptium' -Directory -ErrorAction SilentlyContinue |
          Sort-Object Name -Descending | Select-Object -First 1).FullName
  if (-not $jdk) { throw 'JDK 21 not found. Set JAVA_HOME or install Temurin.' }
  $env:JAVA_HOME = $jdk
}
if (-not $env:MAVEN_HOME) {
  $maven = (Get-ChildItem "$env:USERPROFILE\maven" -Directory -Filter 'apache-maven-*' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1).FullName
  if (-not $maven) {
    $cmd = Get-Command mvn -ErrorAction SilentlyContinue
    if ($cmd) { $maven = Split-Path -Parent (Split-Path -Parent $cmd.Source) }
  }
  if (-not $maven) { throw 'Maven not found. Set MAVEN_HOME or add mvn to PATH.' }
  $env:MAVEN_HOME = $maven
}
if (-not $env:WIX_HOME) {
  $wix = Get-ChildItem "$env:USERPROFILE\wix" -Directory -ErrorAction SilentlyContinue
  if ($wix) { $env:WIX_HOME = "$env:USERPROFILE\wix" }
  else {
    $light = Get-Command light.exe -ErrorAction SilentlyContinue
    if ($light) { $env:WIX_HOME = Split-Path -Parent $light.Source }
  }
  if (-not $env:WIX_HOME) { throw 'WiX not found. Set WIX_HOME or add WiX to PATH.' }
}

$env:PATH = "$env:JAVA_HOME\bin;$env:MAVEN_HOME\bin;$env:WIX_HOME;" + $env:PATH
Set-Location $repo

Write-Host "=== git pull ==="
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'git pull failed (local changes on the Windows host?)' }

Write-Host "=== java ==="; java -version
Write-Host "=== mvn package ==="
mvn -q -B package -DskipTests
if ($LASTEXITCODE -ne 0) { throw 'mvn failed' }

Write-Host "=== staging ==="
Remove-Item -Recurse -Force staging,dist -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force staging | Out-Null
Copy-Item target\openpnp-gui-0.0.1-alpha-SNAPSHOT.jar staging\
Copy-Item -Recurse target\lib staging\lib

$version = (Get-Content VERSION.txt -Raw).Trim() + '.0'
Write-Host "=== jpackage .exe (app-version: $version) ==="
jpackage --type exe --name OpenPnP --app-version $version `
  --description "OpenPnP - Open Source SMT Pick and Place" --vendor "OpenPnP" `
  --input staging --main-jar openpnp-gui-0.0.1-alpha-SNAPSHOT.jar `
  --main-class org.openpnp.Main --dest dist `
  --icon "$repo\src\main\resources\icons\OpenPnP.ico" `
  --java-options "--add-opens=java.base/java.lang=ALL-UNNAMED" `
  --java-options "--add-opens=java.desktop/java.awt=ALL-UNNAMED" `
  --java-options "--add-opens=java.desktop/java.awt.color=ALL-UNNAMED"
if ($LASTEXITCODE -ne 0) { throw 'jpackage .exe failed' }

Write-Host "=== smoke test (app-image launcher) ==="
Remove-Item -Recurse -Force dist-appimage -ErrorAction SilentlyContinue
jpackage --type app-image --name OpenPnP --app-version $version `
  --input staging --main-jar openpnp-gui-0.0.1-alpha-SNAPSHOT.jar `
  --main-class org.openpnp.Main --dest dist-appimage `
  --java-options "--add-opens=java.base/java.lang=ALL-UNNAMED" `
  --java-options "--add-opens=java.desktop/java.awt=ALL-UNNAMED" `
  --java-options "--add-opens=java.desktop/java.awt.color=ALL-UNNAMED"
if ($LASTEXITCODE -ne 0) { throw 'jpackage app-image failed' }
& "$repo\dist-appimage\OpenPnP\OpenPnP.exe" --version
if ($LASTEXITCODE -ne 0) { throw 'smoke test failed' }

Write-Host "=== done ==="
Get-ChildItem dist
