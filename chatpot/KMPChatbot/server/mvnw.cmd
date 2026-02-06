@echo off
rem Maven Wrapper Script for Windows

where mvn >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    mvn %*
) else (
    echo Maven not found. Please install Maven 3.9.5 or higher
    echo Visit: https://maven.apache.org/download.cgi
    exit /b 1
)
