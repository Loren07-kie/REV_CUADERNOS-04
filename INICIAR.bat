@echo off
cd /d "%~dp0"
if not exist .venv py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 goto error
python entrenamiento\entrenar.py
if errorlevel 1 goto error
start http://localhost:5000
python app.py
pause
exit /b
:error
echo No se pudo iniciar. Instala Python 3.11 y revisa el mensaje anterior.
pause
