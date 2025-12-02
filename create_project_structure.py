import os
from pathlib import Path

def create_project_structure(base_path="."):
    folders = [
        "app-ml/entrypoint",
        "app-ml/notebooks",
        "app-ml/src/pipelines",
        "app-ui/assets",
        "common",
        "config",
        "data/raw_data/csv",
        "data/raw_data/parquet",
        "data/prod_data/csv",
        "data/prod_data/parquet",
        "models/experiments",
        "models/prod",
        "images"
    ]
    
    base = Path(base_path)
    
    print("Creating folder structure for Energy Forecast Project...")
    print("-" * 60)
    
    for folder in folders:
        folder_path = base / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")
    
    print("-" * 60)
    print("Folder structure created successfully!")
    print(f"\nBase directory: {base.absolute()}")

if __name__ == "__main__":
    create_project_structure()