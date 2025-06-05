@echo off
REM Optionally activate virtual environment here
REM call "%~dp0venv\Scripts\activate.bat"

REM Run Streamlit from current directory
pushd "%~dp0"
"C:\Users\Shijie\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app.py
popd

pause
