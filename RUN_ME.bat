@echo off
echo ====================================================================================================
echo RFE EXPERIMENTS - FEATURE SELECTION COMPARISON
echo ====================================================================================================
echo.
echo This will run 3 RFE strategies (RFE-only, Hybrid, RFE-CV) with 3 models each.
echo Total: 9 models to train
echo.
echo Estimated time: 20-40 minutes (depending on your CPU)
echo.
echo Progress will be shown with timestamps and percentages.
echo.
pause
echo.
echo Starting training...
echo.

python main.py

echo.
echo ====================================================================================================
echo TRAINING COMPLETE!
echo ====================================================================================================
echo.
echo Check the following directories for results:
echo   - reports_rfe_only/
echo   - reports_hybrid/
echo   - reports_rfe_cv/
echo.
pause
