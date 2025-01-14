# -*- coding: utf-8 -*-
import subprocess
import re
import io
import chardet
import os

# Define the list of project names
project_names = [
    "MyU",
    #"Ai Bot You",
    "Spencer Consulting",
    "Oncoprecisión",
    "Laboratorio Biomed",
    "Trayecto Bookstore",
    "Ortodoncia de la Fuente",
    #"KLIK Muebles",
    #"Nomad Genetics", esta no va
    "House of Spencer"
]
# Path to the parameters.py file


parameters_file_path = "src/parameters.py"

# Function to update the NAME_OF_PROJECT in parameters.py
def update_project_name(parameters_path, new_project_name):
    # Force UTF-8 (ignore chardet or other detection)
    encoding = "utf-8"

    # Read the file with UTF-8
    with open(parameters_path, "r", encoding=encoding) as file:
        lines = file.readlines()
    
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
    
    # Write the updated content back to the file with UTF-8
    with open(parameters_path, "w", encoding=encoding) as file:
        file.writelines(updated_lines)



# Loop through each project name, update parameters.py, and run main.py
for project_name in project_names:
    print(f"Running for project: {project_name}")
    try:
        # Update NAME_OF_PROJECT
        update_project_name(parameters_file_path, project_name)
        
        # Force UTF-8 encoding in the subprocess environment
        subprocess.run(
            ["python", "main.py"],
            check=True,
            text=True,
            encoding="utf-8",
            env=dict(os.environ, PYTHONIOENCODING="utf-8", LANG="C.UTF-8")
        )

    except Exception as e:
        print(f"Error running for project {project_name}: {e}")

print("All projects have been processed.")
