# Wind Power Prediction Project

A reference implementation matching your project description: data cleaning → feature engineering → CEEMDAN (optional) →
feature selection (RFECV) → two models (RandomForest with ASSO; TCN-Attention-BiGRU) → evaluation and outputs.

## Quick start
```bash
# create env (conda recommended)
pip install -r requirements.txt

# run full pipeline (replace with your csv path)
python run.py --csv data/your_data.csv --datetime DATETIME --target POWER
```
