@echo off
chcp 65001>nul
echo ==============================================
echo NEXUS TAURI DESKTOP BUILD (GNU MODE)
echo ==============================================

echo [1/3] Binding Cargo ^& Toolchain Environment...
set CARGO_HOME=D:\cargo_home
set RUSTUP_HOME=C:\Users\p naga babu\.rustup
set "GNU_BIN=C:\ProgramData\mingw64\mingw64\bin"
set "PATH=%GNU_BIN%;%PATH%"

echo [2/3] Building frontend dashboard...
call npm run dashboard:build

echo [3/3] Compiling Tauri Application (Target: GNU)...
call npx @tauri-apps/cli@1 build --target x86_64-pc-windows-gnu

echo ==============================================
echo TAURI DESKTOP COMPILATION DONE.
echo ==============================================
