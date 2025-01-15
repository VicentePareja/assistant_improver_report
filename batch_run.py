# -*- coding: utf-8 -*-
import subprocess
import os

# Define the list of project names
project_names = [
    "MyU",
    # "Ai Bot You",  # Commented project; won't be processed
    "Spencer Consulting",
    "Oncoprecisión",
    "Laboratorio Biomed",
    "Trayecto Bookstore",
    "Ortodoncia de la Fuente",
    # "KLIK Muebles",  # Skipped project
    # "Nomad Genetics",  # Explicitly marked to skip
    "House of Spencer"
]

# Path to the parameters.py file and main.py script
parameters_file_path = "src/parameters.py"
main_script_path = "main.py"

# Log file to capture results
log_file = "run_log.txt"

# Function to validate the existence of critical files
def validate_files():
    if not os.path.exists(parameters_file_path):
        raise FileNotFoundError(f"Parameters file not found: {parameters_file_path}")
    if not os.path.exists(main_script_path):
        raise FileNotFoundError(f"Main script not found: {main_script_path}")

# Function to update the NAME_OF_PROJECT in parameters.py
def update_project_name(parameters_path, new_project_name):
    encoding = "utf-8"  # Use UTF-8 encoding for all file operations

    # Read the parameters.py file
    with open(parameters_path, "r", encoding=encoding) as file:
        lines = file.readlines()

    # Update the NAME_OF_PROJECT line
    updated_lines = []
    name_updated = False
    for line in lines:
        if line.strip().startswith("NAME_OF_PROJECT"):
            updated_lines.append(f'NAME_OF_PROJECT = "{new_project_name}"\n')
            name_updated = True
        else:
            updated_lines.append(line)

    if not name_updated:
        raise ValueError("NAME_OF_PROJECT variable not found in parameters.py")

    # Write the updated content back to the file
    with open(parameters_path, "w", encoding=encoding) as file:
        file.writelines(updated_lines)

# Function to run main.py for the given project
def run_analysis_for_project(project_name):
    try:
        # Update the project name in parameters.py
        update_project_name(parameters_file_path, project_name)

        # Run the main.py script
        subprocess.run(
            ["python", main_script_path],
            check=True,
            text=True,
            encoding="utf-8",
            env=dict(os.environ, PYTHONIOENCODING="utf-8", LANG="C.UTF-8")
        )
        return f"SUCCESS: {project_name}"
    except Exception as e:
        return f"ERROR: {project_name} - {str(e)}"

# Main execution loop
if __name__ == "__main__":
    try:
        # Validate the necessary files exist
        validate_files()

        # Open the log file
        with open(log_file, "w", encoding="utf-8") as log:
            for project_name in project_names:
                print(f"Processing project: {project_name}")
                if project_name.startswith("#") or not project_name.strip():
                    log.write(f"SKIPPED: {project_name}\n")
                    continue

                # Run the analysis for the current project
                result = run_analysis_for_project(project_name)
                log.write(result + "\n")
                print(result)

        print(f"All projects processed. Log written to: {log_file}")
    except Exception as overall_error:
        print(f"Critical error: {overall_error}")
