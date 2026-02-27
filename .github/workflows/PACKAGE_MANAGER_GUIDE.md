# Package Manager Distribution Guide

This guide explains how to distribute Gollek CLI through Homebrew and Chocolatey.

---

## 📦 Homebrew (macOS)

### Setup Homebrew Tap Repository

1. **Create a new repository**: `homebrew-gollek-cli`
   ```bash
   git clone https://github.com/bhangun/homebrew-gollek-cli.git
   ```

2. **Add the Formula**

   Create `Formula/gollek-cli.rb`:
   ```ruby
   class GollekCli < Formula
     desc "Production-ready CLI for Gollek inference platform"
     homepage "https://github.com/bhangun/gollek"
     url "https://github.com/bhangun/gollek/releases/download/gollek-cli-v1.0.0/gollek-cli-macos-x64"
     sha256 "abc123def456..."  # Actual SHA256 from release
     version "1.0.0"
   
     def install
       bin.install "gollek-cli-macos-x64" => "gollek-cli"
     end
   
     def post_install
       puts "🎉 Gollek CLI #{version} installed successfully!"
       puts "Run 'gollek-cli --help' to get started"
     end
   
     test do
       system "#{bin}/gollek-cli", "--version"
     end
   end
   ```

3. **Update with each release**

   Modify the URL and SHA256 hash for each new version.

4. **Users install with**:
   ```bash
   brew tap bhangun/gollek-cli
   brew install gollek-cli
   ```

### Automated Updates via GitHub Actions

Add to your release workflow (the GitHub Actions workflow already generates these):

```yaml
- name: Update Homebrew Formula
  if: startsWith(github.ref, 'refs/tags/gollek-cli-v')
  run: |
    # Update formula file with new version and checksum
    # Commit to homebrew tap repository
```

### Testing Locally

```bash
# Test formula
brew install --build-from-source Formula/gollek-cli.rb

# Verify installation
gollek-cli --version
```

### Publish to Homebrew Core (Optional)

If you want to make it available without tapping:

1. Fork [Homebrew/homebrew-core](https://github.com/Homebrew/homebrew-core)
2. Add formula to `Formula/` directory
3. Submit pull request
4. Homebrew maintainers review and merge

```bash
# Users would then install with
brew install gollek-cli
```

---

## 🍫 Chocolatey (Windows)

### Setup Chocolatey Package

1. **Create package directory**:
   ```
   gollek-cli/
   ├── gollek-cli.nuspec
   └── tools/
       ├── chocolateyinstall.ps1
       ├── chocolateyuninstall.ps1
       └── VERIFICATION.txt
   ```

2. **Create `gollek-cli.nuspec`**:
   ```xml
   <?xml version="1.0" encoding="utf-8"?>
   <package xmlns="http://schemas.microsoft.com/packaging/2015/06/nuspec.xsd">
     <metadata>
       <id>gollek-cli</id>
       <version>1.0.0</version>
       <packageSourceUrl>https://github.com/bhangun/gollek</packageSourceUrl>
       <owners>Kayys Technologies</owners>
       <title>Gollek CLI</title>
       <authors>Kayys Technologies</authors>
       <projectUrl>https://github.com/bhangun/gollek</projectUrl>
       <iconUrl>https://raw.githubusercontent.com/bhangun/gollek/main/inference-gollek/docs/logo.png</iconUrl>
       <licenseUrl>https://github.com/bhangun/gollek/blob/main/LICENSE</licenseUrl>
       <requireLicenseAcceptance>false</requireLicenseAcceptance>
       <projectSourceUrl>https://github.com/bhangun/gollek</projectSourceUrl>
       <docsUrl>https://github.com/bhangun/gollek/wiki</docsUrl>
       <bugTrackerUrl>https://github.com/bhangun/gollek/issues</bugTrackerUrl>
       <tags>gollek cli inference llm ai</tags>
       <summary>Production-ready CLI for Gollek inference</summary>
       <description>
         Gollek CLI is a production-ready command-line interface for running inference 
         with the Gollek platform. Supports local models and cloud providers.
       </description>
       <releaseNotes>https://github.com/bhangun/gollek/releases/tag/gollek-cli-v1.0.0</releaseNotes>
     </metadata>
     <files>
       <file src="tools\**" target="tools" />
     </files>
   </package>
   ```

3. **Create `tools/chocolateyinstall.ps1`**:
   ```powershell
   $ErrorActionPreference = 'Stop'
   
   $toolsDir = "$(Split-Path -parent $MyInvocation.MyCommand.Definition)"
   $url64 = 'https://github.com/bhangun/gollek/releases/download/gollek-cli-v1.0.0/gollek-cli-windows-x64.exe'
   $checksum64 = 'abc123def456...'  # Actual SHA256
   
   $packageArgs = @{
     packageName    = $env:ChocolateyPackageName
     unzipLocation  = $toolsDir
     url64bit       = $url64
     checksum64     = $checksum64
     checksumType64 = 'sha256'
     validExitCodes = @(0)
   }
   
   Get-ChocolateyWebFile @packageArgs
   
   # Rename binary
   $exePath = Join-Path $toolsDir 'gollek-cli-windows-x64.exe'
   if (Test-Path $exePath) {
     Copy-Item $exePath (Join-Path $toolsDir 'gollek-cli.exe')
     Remove-Item $exePath -Force
   }
   ```

4. **Create `tools/chocolateyuninstall.ps1`**:
   ```powershell
   # Nothing needed for uninstall
   ```

5. **Create `tools/VERIFICATION.txt`**:
   ```
   VERIFICATION
   
   Binary verification:
   - SHA256: abc123def456...
   - Download: https://github.com/bhangun/gollek/releases/download/gollek-cli-v1.0.0/gollek-cli-windows-x64.exe
   - Homepage: https://github.com/bhangun/gollek
   ```

### Build and Test Package

```powershell
# Build package
choco pack gollek-cli.nuspec

# Test installation locally
choco install gollek-cli -s . --force

# Test uninstall
choco uninstall gollek-cli
```

### Push to Chocolatey

1. **Register account**: https://chocolatey.org/users/sign-in
2. **Get API key** from account settings
3. **Push package**:
   ```powershell
   choco push gollek-cli.1.0.0.nupkg --key your-api-key
   ```

4. **Wait for moderation** (~1-2 hours)
5. **Users install with**:
   ```powershell
   choco install gollek-cli
   ```

### Automated Updates

The GitHub Actions workflow generates Chocolatey packages automatically. To publish:

```yaml
- name: Push to Chocolatey
  if: startsWith(github.ref, 'refs/tags/gollek-cli-v')
  run: |
    choco push chocolatey-releases/gollek-cli.$VERSION.nupkg \
      --key ${{ secrets.CHOCOLATEY_API_KEY }}
```

---

## 🔄 Automated Release Workflow

The GitHub Actions workflow (`gollek-cli-release.yml`) automatically:

1. ✅ Builds native binaries for all platforms
2. ✅ Generates Homebrew formula with checksums
3. ✅ Generates Chocolatey package with checksums
4. ✅ Creates installation scripts
5. ✅ Uploads as workflow artifacts

### To Complete Distribution

**For Homebrew:**
```bash
# Manual update to homebrew-gollek-cli repository
git clone https://github.com/bhangun/homebrew-gollek-cli.git
# Update Formula/gollek-cli.rb with new version
# Commit and push
```

**For Chocolatey:**
```powershell
# Download generated package from workflow artifacts
choco push gollek-cli.VERSION.nupkg --key $CHOCOLATEY_API_KEY
```

---

## 📊 Distribution Checklist

- [ ] Release tagged and pushed to GitHub
- [ ] GitHub Release created with binaries
- [ ] SHA256SUMS file available in release
- [ ] Homebrew formula updated with new version/checksum
- [ ] Homebrew formula tested locally
- [ ] Homebrew formula pushed to tap repository
- [ ] Chocolatey package updated with new version/checksum
- [ ] Chocolatey package tested locally
- [ ] Chocolatey package pushed to chocolatey.org
- [ ] Installation scripts updated in repository
- [ ] Documentation updated with version information
- [ ] Release notes published on GitHub

---

## 📝 Release Notes Template

```markdown
## Gollek CLI v1.0.0

### Features
- Native compiled binaries (no JVM required)
- Support for local and cloud-based inference
- Fast startup time

### Installation

**Homebrew (macOS):**
```bash
brew tap bhangun/gollek-cli
brew install gollek-cli
```

**Chocolatey (Windows):**
```powershell
choco install gollek-cli
```

**curl (Linux/macOS):**
```bash
curl -fsSL https://raw.githubusercontent.com/bhangun/gollek/main/install-scripts/install.sh | bash
```

### Downloads
- [gollek-cli-linux-x64](#)
- [gollek-cli-macos-x64](#)
- [gollek-cli-windows-x64.exe](#)
- [SHA256SUMS](#)
```

---

## 🔗 Resources

- [Homebrew Formula Documentation](https://docs.brew.sh/Formula-Cookbook)
- [Chocolatey Package Documentation](https://docs.chocolatey.org/)
- [Chocolatey Validator](https://packages.chocolatey.org/viruschecker)
