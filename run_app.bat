@echo off
rem Start the CVD risk screening desktop app (the app can start the local FHIR server itself).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m desktop_app.app
