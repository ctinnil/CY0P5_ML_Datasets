#!/bin/bash

# Check if a directory is provided as a command-line argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

directory="$1"

# Loop through all CSV files in the specified directory
for file in "$directory"/*.csv; do
    if [ -f "$file" ]; then
        echo "File: $file"
        num_columns=$(awk -F ',' '{print NF; exit}' "$file")
        echo "Number of Columns: $num_columns"
        echo  # Add a newline for better readability
    fi
done