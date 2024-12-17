@echo off
setlocal EnableDelayedExpansion

if "%~5"=="" (
    echo Usage: build.bat kokkos_root tutorials_src cpp_compiler build_type backend
    exit /b 1
)

set kokkos_root=%~1
set tutorials_src=%~2
set cpp_compiler=%~3
set build_type=%~4
set backend=%~5

set "EXERCISES=01 02 03"
if "%backend%"=="CUDA" set "EXERCISES=%EXERCISES% 04"

set Kokkos_ROOT=%kokkos_root%
mkdir build 2>nul

for %%e in (%EXERCISES%) do (
    for %%k in (Begin Solution) do (
        set "source_dir=%tutorials_src%\Exercises\%%e\%%k"
        set "build_dir=build\Exercises\%%e\%%k"
        echo building !source_dir!
        cmake -S "!source_dir!" -B "!build_dir!" ^
            -DCMAKE_CXX_COMPILER="%cpp_compiler%" ^
            -DCMAKE_BUILD_TYPE="%build_type%"
        
        cmake --build "!build_dir!" --config "%build_type%"
    )
)
