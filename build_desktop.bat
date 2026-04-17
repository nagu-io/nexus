@echo off
chcp 65001>nul
echo ==============================================
echo NEXUS TAURI DESKTOP BUILD (GNU MODE)
echo ==============================================

echo [1/3] Binding Cargo ^& Toolchain Environment...
rem Use CARGO_HOME / RUSTUP_HOME from the environment if set, otherwise use defaults
if not defined CARGO_HOME set "CARGO_HOME=%USERPROFILE%\.cargo"
if not defined RUSTUP_HOME set "RUSTUP_HOME=%USERPROFILE%\.rustup"
rem Add MinGW to PATH if present in default location
if exist "C:\ProgramData\mingw64\mingw64\bin" (
    set "PATH=C:\ProgramData\mingw64\mingw64\bin;%PATH%"
)

echo [2/3] Building frontend dashboard...
call npm run dashboard:build

echo [3/3] Compiling Tauri Application (Target: GNU)...
call npx @tauri-apps/cli@1 build --target x86_64-pc-windows-gnu

echo ==============================================
echo TAURI DESKTOP COMPILATION DONE.
echo ==============================================
