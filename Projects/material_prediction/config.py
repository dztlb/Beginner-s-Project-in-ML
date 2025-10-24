import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR  = os.path.join(BASE_DIR, "src")

INPUT_XLSX = os.path.join(DATA_DIR, "encoded_data.xlsx")

REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR    = os.path.join(REPORT_DIR, "figs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

LABEL_COL = "Secondary Morphology"

RANDOM_SEED = 42
N_JOBS = -1
CLASS_FOLDS = 5
PRIMARY_SCORING = "accuracy"
FIG_DPI = 140
