# -*- coding: utf-8 -*-

"""
generate_report.py

Generate an HTML report using the summary statistics, correlation matrix,
and best/worst cases produced by analyze_data.py.

Renders data into `report_template.html` (stored in src/templates) and saves
the result to `reports/final_report.html`.

Usage:
    python generate_report.py
"""

import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from src.parameters import NAME_OF_PROJECT


class ReportGenerator:
    """
    A class responsible for generating an HTML report from processed data files.

    Steps:
      1. Load CSV data (summary statistics, correlation matrix, best/worst cases).
      2. Convert DataFrame formats for Jinja2 template rendering.
      3. Render an HTML report from `report_template.html`.
      4. Save the final HTML report to `reports/final_report.html`.
    """

    def __init__(self, base_dir=None):
        """
        Initialize the ReportGenerator.

        Parameters
        ----------
        base_dir : str, optional
            The base directory for the project. If not provided, it is inferred
            from the location of this file.
        """
        # ---------------------------------------------------------------------
        # 1. Set up base paths
        # ---------------------------------------------------------------------
        if base_dir is None:
            # Go two levels up from the location of generate_report.py
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
        self.base_dir = base_dir

        # Data paths
        self.processed_dir = os.path.join(self.base_dir, "data", "processed")
        # Templates directory is inside src/templates
        self.templates_dir = os.path.join(self.base_dir, "src", "templates")
        # The final HTML report goes in `reports/`
        self.reports_dir = os.path.join(self.base_dir, "reports")

        os.makedirs(self.reports_dir, exist_ok=True)

        # ---------------------------------------------------------------------
        # 2. Input and output files
        # ---------------------------------------------------------------------
        self.summary_csv = os.path.join(self.processed_dir, "metrics_summary.csv")
        self.corr_csv = os.path.join(self.processed_dir, "correlation_matrix.csv")
        self.best_worst_csv = os.path.join(self.processed_dir, "best_worst_cases.csv")

        # Template and final output
        self.template_file = "report_template.html"
        self.output_file = os.path.join(self.reports_dir, f"{NAME_OF_PROJECT}final_report.html")

        # ---------------------------------------------------------------------
        # 3. DataFrames (to be loaded in load_data())
        # ---------------------------------------------------------------------
        self.summary_df = None
        self.corr_df = None
        self.best_worst_df = None

        # Data structures for template rendering
        self.summary_data = {}       # dict of lists
        self.best_worst_dict = {}    # nested dict
        self.grade_columns = []      # list of grade column names

    def load_data(self):
        """
        Load CSV files into DataFrames:
          - summary_df: Statistics summary (metrics_summary.csv)
          - corr_df: Correlation matrix (correlation_matrix.csv)
          - best_worst_df: Best and worst cases (best_worst_cases.csv)
        """
        print(f"Loading summary from {self.summary_csv}")
        self.summary_df = pd.read_csv(self.summary_csv)

        print(f"Loading correlation matrix from {self.corr_csv}")
        self.corr_df = pd.read_csv(self.corr_csv, index_col=0)

        print(f"Loading best/worst data from {self.best_worst_csv}")
        self.best_worst_df = pd.read_csv(self.best_worst_csv)

    def prepare_summary_data(self):
        """
        Convert the summary DataFrame into a dictionary of lists for Jinja2,
        and set up self.grade_columns for display.
        """
        # summary_df columns might look like ["metric", "House of Spencer_base_grade", ...]
        self.summary_data = self.summary_df.to_dict(orient="list")

        # Identify grade columns by excluding 'metric'
        self.grade_columns = [c for c in self.summary_data.keys() if c != "metric"]

    def prepare_best_worst_data(self):
        """
        Group the best_worst_df by model_column, separating 'best' and 'worst'
        into a nested dict for each model_column.
        """
        model_columns = self.best_worst_df["model_column"].unique()
        for col in model_columns:
            subset = self.best_worst_df[self.best_worst_df["model_column"] == col]

            # Extract best rows
            best_slice = subset[subset["type"] == "best"].copy()
            best_records = best_slice.to_dict(orient="records")

            # Extract worst rows
            worst_slice = subset[subset["type"] == "worst"].copy()
            worst_records = worst_slice.to_dict(orient="records")

            self.best_worst_dict[col] = {
                "best": best_records,
                "worst": worst_records
            }

    def render_html(self):
        """
        Render the final HTML from the Jinja2 template using the prepared data.
        
        Returns
        -------
        str
            The rendered HTML content.
        """
        env = Environment(loader=FileSystemLoader(self.templates_dir))
        template = env.get_template(self.template_file)

        # We'll pass processed_dir as a *relative path* so images load in the final HTML.
        # In the template, we can reference: {{ processed_dir }}/filename.png
        # Because final_report.html is saved in `reports/`, which is at the same level
        # as `data/processed/`? Actually it’s up one level and over, so we might do "../data/processed"
        # But let's keep it consistent:
        relative_processed_dir = os.path.join("..", "data", "processed")

        html_content = template.render(
            summary_df=self.summary_data,
            corr_df=self.corr_df,
            grade_columns=self.grade_columns,
            processed_dir=relative_processed_dir,
            best_worst_dict=self.best_worst_dict
        )
        return html_content

    def save_html_report(self, html_content):
        """
        Write the rendered HTML to `final_report.html`.

        Parameters
        ----------
        html_content : str
            The full HTML markup to be saved.
        """
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Report generated at: {self.output_file}")

    def run(self):
        """
        Orchestrate the report generation process:
          1. Load data from CSVs.
          2. Prepare summary and best/worst data for Jinja2.
          3. Render HTML.
          4. Save the final report.
        """
        self.load_data()
        self.prepare_summary_data()
        self.prepare_best_worst_data()
        html = self.render_html()
        self.save_html_report(html)


def create_report():
    """
    Main function to run the report generation without direct CLI invocation.
    """
    generator = ReportGenerator()
    generator.run()


if __name__ == "__main__":
    create_report()
