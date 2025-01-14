# -*- coding: utf-8 -*-
import os

# Adjust if these are in src/
from src.analyze_data import DataAnalyzer
from src.generate_report import ReportGenerator
from src.parameters import (
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    GRADE_COLUMNS,
    RESPONSE_MAP,
    NBESTWORST
)


def main():
    """
    Main entry point for:
      1) Running the analysis (generating CSV/plots in data/processed)
      2) Generating the HTML report into reports/final_report.html
    """

    # 1. Run the analysis
    analyzer = DataAnalyzer(
        input_file=RAW_DATA_FILE,
        output_dir=PROCESSED_DATA_DIR,
        grade_columns=GRADE_COLUMNS,
        response_map=RESPONSE_MAP
    )
    analyzer.run_analysis(n_best_worst=NBESTWORST)

    # 2. Generate the HTML report
    #    We'll explicitly set base_dir to the root "assistant_improver_report"
    #    so that 'generate_report.py' looks in the correct places.
    #    e.g., ".../assistant_improver_report"
    base_dir = os.path.abspath(os.path.dirname(__file__))  # The folder where this main.py is

    report = ReportGenerator(base_dir=base_dir)
    # The run() method will load:
    #   data/processed/metrics_summary.csv,
    #   data/processed/correlation_matrix.csv,
    #   data/processed/best_worst_cases.csv
    # and render final_report.html to the `reports` folder.
    report.run()


if __name__ == "__main__":
    main()
