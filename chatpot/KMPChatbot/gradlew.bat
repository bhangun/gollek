@rem Gradle startup script for Windows

@if "%DEBUG%" == "" @echo off
setlocal enabledelayedexpansion

set DIRNAME=%~dp0
if "%DIRNAME%" == "" set DIRNAME=.
set APP_BASE_NAME=%~n0
set GRADLE_USER_HOME=%USERPROFILE%\.gradle

echo Starting Gradle...
echo Please ensure Gradle 8.2+ is installed
echo Visit: https://gradle.org/install/

if exist "%GRADLE_USER_HOME%\wrapper\dists\gradle-8.2-bin" (
    call gradle %*
) else (
    echo Gradle wrapper not found. Please install Gradle 8.2+
    exit /b 1
)
