"""Configuration file containing global variables"""
from pathlib import Path


# FILE SYSTEM LOCATIONS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"
REPORT_DIR      = PROJECT_ROOT / "reports"

# RUNTIME PARAMETERS
RANDOM_SEED  = 42
TEST_SIZE    = 0.20

# TRAINING PARAMETERS
NUM_FEATURES = 100
