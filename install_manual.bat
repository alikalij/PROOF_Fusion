@echo off
echo Installing requirements for PROOF project...

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo.
echo ========================================
echo Installing PyTorch...
echo ========================================
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo ========================================
echo Installing other packages...
echo ========================================
pip install tqdm matplotlib numpy

echo.
echo ========================================
echo Installing open-clip...
echo ========================================
pip install open-clip

echo.
echo ========================================
echo Installing timm...
echo ========================================
pip install timm

echo.
echo ========================================
echo Installation completed!
echo ========================================
pause 