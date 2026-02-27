# Gollek CLI Release Workflow

## Trigger

```bash
git tag 0.1.0
git push origin 0.1.0
```

Workflow: `.github/workflows/golek-cli-release.yml`

## What It Publishes

The release now includes:

- Native binaries:
  - `gollek-linux-x64`
  - `gollek-macos-x64`
  - `gollek-windows-x64.exe`
- Installer/archives:
  - `gollek-linux-x64.tar.gz`
  - `gollek-macos-x64.tar.gz`
  - `gollek-macos-x64.pkg`
  - `gollek-windows-x64.zip`
- JVM fallback bundle:
  - `gollek-jvm.zip` (Windows/macOS/Linux with Java 21+)
- Installer scripts:
  - `install.sh` (curl install path)
  - `install.ps1` (PowerShell install path)
- Package manager templates:
  - `gollek.rb` (Homebrew formula)
  - `gollek-chocolatey-template.zip` (Chocolatey nuspec + tools)
- `SHA256SUMS`

## Installation Options

### curl (macOS/Linux)

```bash
curl -fsSL https://github.com/bhangun/gollek/releases/latest/download/install.sh | bash
```

### Homebrew (tap formula)

Use the generated `gollek.rb` in your tap repo, then:

```bash
brew tap bhangun/gollek
brew install gollek
```

### Chocolatey

Use `gollek-chocolatey-template.zip` to build/publish package, then:

```powershell
choco install gollek
```

### Windows Native (direct)

```powershell
Invoke-WebRequest -Uri "https://github.com/bhangun/gollek/releases/latest/download/gollek-windows-x64.exe" -OutFile "gollek.exe"
.\gollek.exe --version
```

### Windows JVM fallback

Download `gollek-jvm.zip`, extract, then run:

```powershell
.\bin\gollek.bat --version
```
