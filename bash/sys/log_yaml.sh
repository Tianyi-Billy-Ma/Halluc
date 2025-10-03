#!/bin/bash

# Script to display YAML file contents
# Usage: ./log_yaml.sh yaml_file_path


YAML_FILE="$1"

cd $WORK_DIR

# Check if file exists
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: YAML file '$YAML_FILE' not found!"
    echo "Available YAML files in configs directory:"
    find "$WORK_DIR/configs" -name "*.yaml" -o -name "*.yml" | head -10
    exit 1
fi

echo "================================================"
echo "Displaying YAML file: $YAML_FILE"
echo "================================================"

# Display the YAML file using cat
cat "$YAML_FILE"

echo ""
echo "================================================"
echo "End of YAML file: $YAML_FILE"
echo "================================================"

