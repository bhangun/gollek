# Gollek CLI Native Release Guide

This document explains how to create and manage native compiled releases of Gollek CLI for macOS, Windows, and Linux.

## Overview

The `gollek-cli-release.yml` GitHub Actions workflow automatically builds native executables for:
- **Linux** (x86_64)
- **macOS** (x86_64)
- **Windows** (x86_64)

Native executables are compiled using GraalVM Native Image, providing:
- ✅ Single executable (no JVM needed)
- ✅ Fast startup time (~100ms)
- ✅ Reduced memory footprint
- ✅ Cross-platform support

## Triggering a Release

### Method 1: Git Tag (Automated)

```bash
# Create and push a tag
git tag gollek-cli-v1.0.0
git push origin gollek-cli-v1.0.0
```

The workflow will automatically trigger and create a GitHub Release with native binaries.

### Method 2: Manual Trigger

Go to GitHub Actions → Gollek CLI Native Release → Run workflow manually.

## Workflow Details

### Build Process

For each platform:
1. **Setup Java** - Installs GraalVM JDK 25
2. **Install Dependencies** - Platform-specific build tools
3. **Build Native** - Compiles with `mvn clean package -Pnative -DskipTests`
4. **Upload Artifacts** - Stores compiled binaries

### Build Stages

```
build-native (parallel on 3 OS)
    ↓
prepare-release (consolidates artifacts)
    ↓
release (creates GitHub Release)
```

### Output Artifacts

After successful build, artifacts are available as:
- `gollek-cli-linux-x64` - Linux executable
- `gollek-cli-macos-x64` - macOS executable
- `gollek-cli-windows-x64.exe` - Windows executable
- `SHA256SUMS` - Checksum file for verification

## Using Released Binaries

### Linux

```bash
# Download
wget https://github.com/bhangun/gollek/releases/download/gollek-cli-v1.0.0/gollek-cli-linux-x64

# Make executable
chmod +x gollek-cli-linux-x64

# Run
./gollek-cli-linux-x64 --help
```

### macOS

```bash
# Download
curl -L -O https://github.com/bhangun/gollek/releases/download/gollek-cli-v1.0.0/gollek-cli-macos-x64

# Make executable
chmod +x gollek-cli-macos-x64

# Run (may need to approve in security settings)
./gollek-cli-macos-x64 --help
```

### Windows

```powershell
# Download (using PowerShell)
Invoke-WebRequest -Uri "https://github.com/bhangun/gollek/releases/download/gollek-cli-v1.0.0/gollek-cli-windows-x64.exe" -OutFile "gollek-cli.exe"

# Run
.\gollek-cli.exe --help
```

## Installation Scripts

### Linux/macOS Install Script

```bash
#!/bin/bash
VERSION="1.0.0"
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH="x64"

if [ "$OS" = "darwin" ]; then
    OS="macos"
fi

curl -L "https://github.com/bhangun/gollek/releases/download/gollek-cli-v${VERSION}/gollek-cli-${OS}-${ARCH}" \
    -o gollek-cli

chmod +x gollek-cli
sudo mv gollek-cli /usr/local/bin/

echo "Gollek CLI $VERSION installed successfully!"
gollek-cli --version
```

### Verify Checksum

```bash
# Download checksums
wget https://github.com/bhangun/gollek/releases/download/gollek-cli-v1.0.0/SHA256SUMS

# Verify
sha256sum -c SHA256SUMS
```

## Build Requirements

### Linux
- GCC/Make
- OpenSSL development files
- Standard build tools

### macOS
- Xcode Command Line Tools
- Automake (installed via Homebrew)

### Windows
- Visual C++ Build Tools
- Windows SDK

All are handled automatically by the GitHub Actions workflow.

## Customization

### Modify Java Version

Edit `.github/workflows/gollek-cli-release.yml`:

```yaml
- name: Setup Java
  uses: actions/setup-java@v4
  with:
    distribution: graalvm
    java-version: '24'  # Change version here
```

### Add Pre-release Versions

Modify tag pattern matching:

```yaml
on:
  push:
    tags:
      - 'gollek-cli-v*'
      - 'gollek-cli-*-rc*'  # Add RC versions
```

### Include Source Files

Add to `prepare-release` job:

```bash
# Create source archive
cd inference-gollek/ui/gollek-cli
tar -czf ../../release/gollek-cli-sources.tar.gz src/
```

## Troubleshooting

### Build Fails on macOS

Ensure Xcode Command Line Tools are up to date:

```bash
xcode-select --install
brew install automake
```

### Build Fails on Linux

Check GraalVM installation and required libraries:

```bash
apt-get install -y build-essential libssl-dev zlib1g-dev pkg-config
```

### Windows Build Issues

Ensure Visual C++ Build Tools are installed:
- Download from: https://visualstudio.microsoft.com/downloads/
- Select "Desktop development with C++"

## Performance Metrics

Typical build times (from GitHub Actions):
- **Linux**: 8-10 minutes
- **macOS**: 12-15 minutes
- **Windows**: 15-20 minutes

Total workflow time: ~25-30 minutes (parallel builds)

## Rollback

If a release has issues:

```bash
# Delete the tag locally and remote
git tag -d gollek-cli-v1.0.0
git push origin --delete gollek-cli-v1.0.0

# Delete GitHub Release via web UI
# Re-tag and push when ready
```

## Next Steps

1. Configure GitHub Release notes (optional)
2. Set up artifact retention policies
3. Add Homebrew/Chocolatey distribution (future enhancement)
4. Implement auto-update mechanism for CLI
