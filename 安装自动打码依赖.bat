@echo off
chcp 65001 >nul
setlocal
echo Forge Neo 图片工坊 - 自动打码可选依赖
echo.
echo 将保留 Forge Neo 当前 NumPy，不执行主环境降级。
echo.
set "FORGE_ROOT=%~1"
if "%FORGE_ROOT%"=="" set /p "FORGE_ROOT=请输入 Forge Neo 根目录（包含 launch.py）: "
if "%FORGE_ROOT%"=="" goto :error
if not exist "%FORGE_ROOT%\launch.py" goto :error
if not exist "%FORGE_ROOT%\venv\Scripts\python.exe" goto :error
"%FORGE_ROOT%\venv\Scripts\python.exe" -m pip install --no-deps -r "%~dp0requirements-censor.txt"
if errorlevel 1 goto :error
echo.
echo 安装完成。重启 Forge Neo 后打开“图片工坊”，使用“自动打码”区。
pause
exit /b 0
:error
echo 无法确认 Forge Neo 根目录。请传入包含 launch.py 和 venv\Scripts\python.exe 的目录。
pause
exit /b 1
