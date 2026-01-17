@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Running Training Pipeline...
python train_model.py

echo Starting Dashboard...
streamlit run dashboard/app.py
pause
