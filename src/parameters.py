# -*- coding: utf-8 -*-
import os

NAME_OF_PROJECT = "House of Spencer"

# Project root directory (up one level from 'src')
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Raw CSV file (input)
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, f"{NAME_OF_PROJECT}_unified_results.csv")

# ------------------------------------------------------------------------------
# Grade Columns and Response Map
# ------------------------------------------------------------------------------
# These columns should exist in your CSV file and contain numeric scores
GRADE_COLUMNS = [
    f"{NAME_OF_PROJECT}_base_grade",
    f"{NAME_OF_PROJECT}_fine_tuned_grade",
]

# Map each grade column to its corresponding machine response column
RESPONSE_MAP = {
    f"{NAME_OF_PROJECT}_base_grade": f"{NAME_OF_PROJECT}_base_answer",
    f"{NAME_OF_PROJECT}_fine_tuned_grade": f"{NAME_OF_PROJECT}_fine_tuned_answer",
}

# Number of best/worst samples to retrieve in the analysis
NBESTWORST = 3

# ------------------------------------------------------------------------------
# Plotting Parameters
# ------------------------------------------------------------------------------
# Score axis limits (for histograms and boxplots)
X_MIN = 0
X_MAX = 5
