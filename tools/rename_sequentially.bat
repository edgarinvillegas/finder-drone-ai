@echo off
set /p start=Please enter the starting control number: 

setlocal enableDelayedExpansion

for /r %%g in (*.jpg) do (call :RenameIt %%g)
goto :eof
goto :exit

:RenameIt
echo Renaming "%~nx1" to !start!%~x1
ren "%~nx1" !start!%~x1
set /a start+=1
goto :eof

:exit
exit /b