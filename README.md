# Machine-Learning
# Create Project Starter
#!/bin/bash

# Check if project name is provided
```
if [ -z "$1" ]; then
    echo "Usage: ./create_ml_app.sh <project_name>"
    exit 1
fi

PROJECT_NAME=$1
```
# Create main project folder
```
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME" || exit
```
# Create folders
```
mkdir -p data notebooks
mkdir -p src/api src/data src/features src/models src/inference src/utils src/visualization
mkdir -p configs scripts tests
```
# Create placeholder files
```
touch src/api/main.py
touch requirements.txt Dockerfile README.md .gitignore
touch configs/config.yaml
touch scripts/run_training.sh scripts/run_inference.sh
```
# Optional: create empty Python files for modules
```
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/inference/__init__.py
touch src/utils/__init__.py
touch src/visualization/__init__.py
touch tests/test_models.py

echo "Project structure '$PROJECT_NAME' created successfully!"
```

# Save this as 
create_ml_app.sh. 


Make it executable:
```
chmod +x create_ml_app.sh
```

Run the script with your project name:
```
./create_ml_app.sh my_ai_project
```
