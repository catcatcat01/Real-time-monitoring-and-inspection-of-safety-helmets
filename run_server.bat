@echo off
echo Starting Waitress Server...
waitress-serve --listen=10.60.208.45:5000 app1:app
pause