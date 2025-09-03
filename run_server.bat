@echo off
echo Starting Waitress Server...
waitress-serve --listen=0.0.0.0:8080 app1:app

pause
