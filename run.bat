@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "CMD=%~1"
if "%CMD%"=="" set "CMD=app"

if /I "%CMD%"=="setup" goto :setup
if /I "%CMD%"=="help" goto :help

if not exist ".venv\Scripts\python.exe" (
    echo [!] Environment not found. Run:  run.bat setup
    exit /b 1
)
call ".venv\Scripts\activate.bat"

if /I "%CMD%"=="app" goto :app
if /I "%CMD%"=="cli" goto :cli
if /I "%CMD%"=="examples" goto :examples
if /I "%CMD%"=="warmup" goto :warmup
if /I "%CMD%"=="train" goto :train
if /I "%CMD%"=="test" goto :test
goto :help

:setup
set "PYEXE=python"
where py >nul 2>nul
if not errorlevel 1 (
    py -3.10 --version >nul 2>nul
    if not errorlevel 1 set "PYEXE=py -3.10"
)
echo Using interpreter: %PYEXE%
%PYEXE% -m venv .venv || goto :fail
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip wheel setuptools || goto :fail
nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [warn] No NVIDIA GPU detected - installing CPU PyTorch. Generation needs a GPU or Google Colab.
    pip install torch==2.3.1 torchvision==0.18.1 || goto :fail
) else (
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 || goto :fail
)
pip install -r requirements.txt || goto :fail
if not exist "models\TripoSR" (
    git clone https://github.com/VAST-AI-Research/TripoSR.git models\TripoSR || goto :fail
)
python create_examples.py
echo.
echo Setup finished. Next:  run.bat warmup   (downloads about 10 GB of model weights)   then   run.bat app
goto :end

:app
shift
python app.py %1 %2 %3 %4 %5 %6 %7 %8 %9
goto :end

:cli
shift
python cli.py %1 %2 %3 %4 %5 %6 %7 %8 %9
goto :end

:examples
python create_examples.py
goto :end

:warmup
python cli.py --warmup
goto :end

:train
shift
python train.py %1 %2 %3 %4 %5 %6 %7 %8 %9
goto :end

:test
python -m pytest -q tests
goto :end

:help
echo.
echo   run.bat setup      create .venv, install PyTorch + dependencies, clone TripoSR
echo   run.bat warmup     download and load all model weights once
echo   run.bat app        start the web app            (add --share for a public link)
echo   run.bat cli ...    command line, e.g.  run.bat cli examples\mug.png --prompt "red mug"
echo   run.bat examples   regenerate the example sketches
echo   run.bat train      train the optional QuickDraw classifier (add --quick for a smoke test)
echo   run.bat test       run the unit tests
echo.
goto :end

:fail
echo [x] Command failed - see the message above.
exit /b 1

:end
endlocal
