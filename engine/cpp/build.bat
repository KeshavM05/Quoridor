@echo off
REM Build script for Windows (MSVC or MinGW)
REM Prerequisites: pip install pybind11 numpy, CMake >= 3.16, Visual Studio or MinGW

setlocal
set SCRIPT_DIR=%~dp0
set ENGINE_DIR=%SCRIPT_DIR%..

echo === Building C++ Quoridor Engine ===
echo Source: %SCRIPT_DIR%
echo Output: %ENGINE_DIR%

REM Ensure pybind11 is installed
python -c "import pybind11" 2>nul || (
    echo Installing pybind11...
    pip install pybind11
)

REM Create build directory
if not exist "%SCRIPT_DIR%build" mkdir "%SCRIPT_DIR%build"
cd /d "%SCRIPT_DIR%build"

REM Configure
echo.
echo --- Configuring with CMake ---
cmake .. -DCMAKE_BUILD_TYPE=Release

REM Build
echo.
echo --- Compiling ---
cmake --build . --config Release

REM Copy the built module
echo.
echo --- Installing ---
for /r %%f in (quoridor_cpp*.pyd) do (
    copy "%%f" "%ENGINE_DIR%\" /y
    echo Copied %%~nxf to %ENGINE_DIR%
    goto :done_copy
)
for /r %%f in (quoridor_cpp*.so) do (
    copy "%%f" "%ENGINE_DIR%\" /y
    echo Copied %%~nxf to %ENGINE_DIR%
    goto :done_copy
)
echo ERROR: Could not find built module!
exit /b 1

:done_copy
echo.
echo === Build complete! ===
echo.
echo Test it:
echo   cd %ENGINE_DIR%
echo   python -c "import quoridor_cpp; g = quoridor_cpp.QuoridorGame(); print(g)"
