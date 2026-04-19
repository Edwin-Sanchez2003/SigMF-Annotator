@echo off
:: Build script for SigMF Waterfall Annotator
:: Run this from the project root on a Windows machine.
::
:: Prerequisites (run once):
::   pip install pyinstaller
::   pip install -e .

echo === SigMF Waterfall Annotator - Windows Build ===

:: Clean previous build artefacts
if exist build     rmdir /s /q build
if exist dist      rmdir /s /q dist

:: Run PyInstaller with the spec file
pyinstaller sigmf_annotator.spec

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller failed. See output above.
    exit /b 1
)

echo.
echo === Build complete ===
echo Output: dist\sigmf_annotator\sigmf_annotator.exe
