@echo off
echo =========================================
echo    Starting CrimeScope Dashboard
echo =========================================

echo.
echo [1/3] Starting Python Backend API (Port 5000)...
start "CrimeScope Backend" cmd /k "python backend/app.py"

echo [2/3] Starting Local Web Server (Port 8000)...
start "CrimeScope Frontend" cmd /k "python -m http.server 8000"

echo [3/3] Opening Dashboard in your Web Browser...
timeout /t 5 >nul
start http://127.0.0.1:8000/CrimeScope-Web/index.html

echo.
echo Dashboard is now running! 
echo Keep the two black command prompt windows open while using the site.
echo You can safely close this window.
pause
