@echo off

set PYGEM_DIR=%~dp0
set GEMMULE=%PYGEM_DIR%..\..\



mkdir %PYGEM_DIR%\mk-build
pushd %PYGEM_DIR%
cd %PYGEM_DIR%\mk-build
cmake %GEMMULE%


cmake --build . --config Release
cmake --install . --config Release --prefix %PYGEM_DIR%\mk-build

popd
set INCLUDE_DIR=%PYGEM_DIR%\mk-build\include
set LIB_DIR=%PYGEM_DIR%\mk-build\lib
python -m build

rem && set LDFLAGS="-L%PYGEM_DIR%\mk-build\lib" && python -m build


for %%i in (dist\*.whl) do pip install %%i

rem pip install dist\*.whl
