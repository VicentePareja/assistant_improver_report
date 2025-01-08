# -*- coding: utf-8 -*-

"""
analyze_data.py

This script defines a class (DataAnalyzer) that loads the CSV data (as
configured in parameters.py), performs analysis (descriptive statistics,
correlations, best/worst cases), creates visualizations, and saves the
results to disk.

Usage:
    Called by main.py, or run directly (not typical).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Import parameters from parameters.py
# Adjust if needed to match your folder structure
from src.parameters import (
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    GRADE_COLUMNS,
    RESPONSE_MAP,
    NBESTWORST,
    X_MIN,
    X_MAX
)


class DataAnalyzer:
    """
    Encapsulates data analysis functions, including:
      - Data loading and cleaning
      - Computing descriptive statistics
      - Identifying best/worst cases
      - Correlation analysis
      - Saving results and creating plots
    """

    def __init__(self, input_file, output_dir, grade_columns, response_map):
        """
        Parameters
        ----------
        input_file : str
            Path to the input CSV file.
        output_dir : str
            Path where processed outputs (CSV, plots) will be saved.
        grade_columns : list of str
            Columns representing numeric scores/grades.
        response_map : dict
            Mapping from each grade column to its model response column.
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.grade_columns = grade_columns
        self.response_map = response_map
        self.df = pd.DataFrame()

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        Loads the data from CSV into self.df and coerces grade columns to numeric.
        """
        print(f"Loading data from: {self.input_file}")
        self.df = pd.read_csv(self.input_file)

        # Convert grade columns to numeric
        for col in self.grade_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def summarize_grades(self):
        """
        Computes descriptive statistics for each grade column.
        Returns a dict with stats for each column.
        """
        stats_dict = {}
        for col in self.grade_columns:
            valid_data = self.df[col].dropna()
            if len(valid_data) == 0:
                stats_dict[col] = {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "25%": None,
                    "50%": None,
                    "75%": None,
                    "max": None
                }
            else:
                stats_dict[col] = {
                    "count": len(valid_data),
                    "mean": valid_data.mean(),
                    "std": valid_data.std(),
                    "min": valid_data.min(),
                    "25%": valid_data.quantile(0.25),
                    "50%": valid_data.quantile(0.50),  # median
                    "75%": valid_data.quantile(0.75),
                    "max": valid_data.max()
                }
        return stats_dict

    def find_best_worst(self, n=3):
        """
        Finds top n (best) and bottom n (worst) rows based on each grade column.
        Returns a dict with sub-dicts for 'best' and 'worst' DataFrames.
        """
        best_worst_dict = {}
        for col in self.grade_columns:
            # Sort ascending to find worst
            sorted_df = self.df.sort_values(by=col, ascending=True)
            worst_n = sorted_df.dropna(subset=[col]).head(n)

            # Sort descending for best
            best_n = self.df.dropna(subset=[col]).nlargest(n, col)

            best_worst_dict[col] = {
                "best": best_n,
                "worst": worst_n
            }
        return best_worst_dict

    def compute_top_bottom_averages(self, n=3):
        """
        For each grade column, compute the average of the top n and bottom n values.
        Returns a dict with bottom_n_avg and top_n_avg for each column.
        """
        top_bottom_dict = {}
        for col in self.grade_columns:
            valid_data = self.df.dropna(subset=[col]).copy()
            valid_data_sorted = valid_data.sort_values(by=col, ascending=True)

            bottom_n_avg = valid_data_sorted.head(n)[col].mean() if len(valid_data_sorted) >= n else np.nan
            top_n_avg = valid_data_sorted.tail(n)[col].mean() if len(valid_data_sorted) >= n else np.nan

            top_bottom_dict[col] = {
                "bottom_n_avg": bottom_n_avg,
                "top_n_avg": top_n_avg
            }
        return top_bottom_dict

    def correlation_analysis(self):
        """
        Computes the correlation matrix for the grade columns and returns it as a DataFrame.
        """
        return self.df[self.grade_columns].corr()

    def save_descriptive_statistics(self, stats_dict, output_file):
        """
        Saves descriptive statistics (including percentiles) to a CSV file.
        """
        metrics_order = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        summary_data = {"metric": metrics_order}

        for col in self.grade_columns:
            col_stats = stats_dict[col]
            summary_data[col] = [
                col_stats["count"],
                col_stats["mean"],
                col_stats["std"],
                col_stats["min"],
                col_stats["25%"],
                col_stats["50%"],
                col_stats["75%"],
                col_stats["max"]
            ]

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_file, index=False)
        print(f"Summary statistics saved to: {output_file}")

    def save_correlation_matrix(self, corr_df, output_file):
        """
        Saves the correlation matrix to a CSV file.
        """
        corr_df.to_csv(output_file)
        print(f"Correlation matrix saved to: {output_file}")

    def save_best_worst_cases(self, best_worst_dict, output_file):
        """
        Saves best/worst cases for each grade column into a CSV file.
        Fetches the machine responses from the correct mapped columns.
        """
        bw_rows = []
        for col in self.grade_columns:
            # Determine the corresponding machine response column
            model_col_name = self.response_map.get(col, None)

            if not model_col_name:
                print(f"Warning: No response column mapped for {col}. Skipping machine responses.")
                continue

            # Get "best" and "worst" data
            best_df = best_worst_dict[col]["best"]
            worst_df = best_worst_dict[col]["worst"]

            # Process "best" cases
            for _, row in best_df.iterrows():
                bw_rows.append({
                    "type": "best",
                    "model_column": col,
                    "grade": row[col],
                    "question": row.get("question", None),
                    "human_response": row.get("human_response", None),
                    "machine_response": row.get(model_col_name, "N/A")  # Fetch machine response
                })

            # Process "worst" cases
            for _, row in worst_df.iterrows():
                bw_rows.append({
                    "type": "worst",
                    "model_column": col,
                    "grade": row[col],
                    "question": row.get("question", None),
                    "human_response": row.get("human_response", None),
                    "machine_response": row.get(model_col_name, "N/A")  # Fetch machine response
                })

        # Convert to DataFrame and save to CSV
        bw_df = pd.DataFrame(bw_rows)
        bw_df.to_csv(output_file, index=False)
        print(f"Best/worst cases saved to: {output_file}")



    def create_distribution_plots(self, x_min=0, x_max=5):
        """
        Creates histogram plots for each grade column (without KDE) and saves them to PNG files.
        Includes a vertical line for the mean.
        """
        for col in self.grade_columns:
            plt.figure(figsize=(6, 4))
            # Histogram only, without KDE
            sns.histplot(self.df[col], kde=False, bins=10, color='blue')
            plt.xlim(x_min, x_max)

            # Plot the mean as a vertical line
            mean_val = self.df[col].mean()
            if not np.isnan(mean_val):
                plt.axvline(x=mean_val, color='red', linestyle='--', label=f'Mean={mean_val:.2f}')

            plt.title(f"Histogram of {col}")
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()

            output_path = os.path.join(self.output_dir, f"{col}_hist.png")
            plt.savefig(output_path, dpi=100)
            plt.close()
            print(f"Histogram saved: {output_path}")


    def create_boxplots(self, y_min=0, y_max=5):
        """
        Creates boxplots for each grade column using Plotly and saves them to an HTML file.
        """
        valid_data = self.df[self.grade_columns].dropna(how='all')  # Drop rows with all NaNs
        if not valid_data.empty:
            fig = go.Figure()

            # Add box plots for each grade column
            for col in self.grade_columns:
                fig.add_trace(go.Box(
                    y=valid_data[col],
                    name=col,
                    boxmean=True  # Show mean on the boxplot
                ))

            # Update layout for better appearance
            fig.update_layout(
                title="Box Plot of Grades",
                yaxis=dict(
                    title="Score",
                    range=[y_min, y_max]
                ),
                xaxis=dict(
                    title="Grade Columns"
                ),
                template="plotly_white"
            )

            # Save the plot to an HTML file
            output_path = os.path.join(self.output_dir, "grades_boxplot.html")
            fig.write_html(output_path)
            print(f"Boxplot saved: {output_path}")
            
        else:
            print("No valid numeric data found for boxplots. Skipping boxplot creation.")

    def create_correlation_heatmap(self, corr_df):
        """
        Creates a heatmap of the correlation matrix and saves it to a PNG file.
        """
        plt.figure(figsize=(5, 4))
        sns.heatmap(corr_df, annot=True, cmap="Blues", vmin=-1, vmax=1, square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"Correlation heatmap saved: {output_path}")

    def run_analysis(self, n_best_worst=3):
        """
        Full pipeline:
          1. Load data
          2. Descriptive statistics
          3. Best/worst entries
          4. Correlation
          5. Visualization
          6. Save everything
        """
        # 1. Load
        self.load_data()

        # 2. Descriptive Stats
        stats_dict = self.summarize_grades()

        # 3. Best/Worst
        best_worst_dict = self.find_best_worst(n=n_best_worst)

        # 4. Top/Bottom averages
        top_bottom_dict = self.compute_top_bottom_averages(n=n_best_worst)

        # 5. Correlation
        corr_df = self.correlation_analysis()

        # Print summary to console
        print("\n===== DESCRIPTIVE STATISTICS =====")
        for col in self.grade_columns:
            col_stats = stats_dict[col]
            print(f"\nColumn: {col}")
            print(f"  Count: {col_stats['count']}")
            print(f"  Mean:  {col_stats['mean']}")
            print(f"  Std:   {col_stats['std']}")
            print(f"  Min:   {col_stats['min']}")
            print(f"  25%:   {col_stats['25%']}")
            print(f"  50%:   {col_stats['50%']}")
            print(f"  75%:   {col_stats['75%']}")
            print(f"  Max:   {col_stats['max']}")

            print(f"  --> Bottom {n_best_worst} Avg: {top_bottom_dict[col]['bottom_n_avg']}")
            print(f"  --> Top {n_best_worst} Avg: {top_bottom_dict[col]['top_n_avg']}")

        print("\n===== CORRELATION MATRIX =====")
        print(corr_df)

        # 6. Save outputs
        # Descriptive stats
        summary_file = os.path.join(self.output_dir, "metrics_summary.csv")
        self.save_descriptive_statistics(stats_dict, summary_file)

        # Correlation
        corr_file = os.path.join(self.output_dir, "correlation_matrix.csv")
        self.save_correlation_matrix(corr_df, corr_file)

        # Best/Worst
        bw_file = os.path.join(self.output_dir, "best_worst_cases.csv")
        self.save_best_worst_cases(best_worst_dict, bw_file)

        # Plots
        self.create_distribution_plots(x_min=X_MIN, x_max=X_MAX)
        self.create_boxplots(y_min=X_MIN, y_max=X_MAX)
        self.create_correlation_heatmap(corr_df)

        print("\nAnalysis complete. All results are saved to:", self.output_dir)


def main():
    """
    Optional: If you run this file directly, we'll execute the analysis.
    Typically, you'd call this from a separate 'main.py' or CLI script.
    """
    analyzer = DataAnalyzer(
        input_file=RAW_DATA_FILE,
        output_dir=PROCESSED_DATA_DIR,
        grade_columns=GRADE_COLUMNS,
        response_map=RESPONSE_MAP
    )
    analyzer.run_analysis(n_best_worst=NBESTWORST)


if __name__ == "__main__":
    main()
