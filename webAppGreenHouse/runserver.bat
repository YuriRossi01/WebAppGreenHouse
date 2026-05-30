@echo off
setlocal

:: 1. Torno indietro di due cartelle per trovare la cartella venv (IdeaProjects/venv)
set "VENV_PATH=%~dp0..\venv"

echo [1/3] Attivazione ambiente virtuale in venv
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [ERRORE] Non trovo la venv a quel percorso!
    pause
    exit /b
)

call "%VENV_PATH%\Scripts\activate.bat"

:: 2. Entro nella sottocartella dove c'e' manage.py
echo [2/3] Avvio server Django...
cd /d "%~dp0mysite"

:: Lancio il server in background
start /b python manage.py runserver

:: 3. Ciclo di attesa (FIXED SYNTAX)
echo [3/3] In attesa che il server risponda...
:waitForServer
timeout /t 2 >nul

curl -s http://127.0.0.1:8000 >nul
if %ERRORLEVEL% NEQ 0 (
    echo ...il server non e' ancora pronto, riprovo...
    goto waitForServer
)
:: Messaggio di successo, apertura browser
echo [OK] Server attivo! Apertura Chrome...
start chrome http://127.0.0.1:8000/home/