@echo off
echo 正在启动LLaMA-Factory...
echo.

:: 进入项目目录，根据实际路径修改
cd /d %~dp0

:: 检查Python环境
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo 未检测到Python，请确保已安装Python并添加到环境变量中。
    pause
    exit /b
)

:: 检查是否有虚拟环境，如果有则激活
if exist venv\Scripts\activate.bat (
    echo 检测到虚拟环境，正在激活...
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo 检测到虚拟环境，正在激活...
    call .venv\Scripts\activate.bat
)

:: 启动LLaMA-Factory
python -m llamafactory webui

:: 如果启动失败，可以尝试以下备用命令之一
:: python -m src.llamafactory.webui.cli
:: python -m src.llamafactory.webui.app

echo.
echo 如果启动成功，请在浏览器中访问: http://localhost:7860
echo.
pause 