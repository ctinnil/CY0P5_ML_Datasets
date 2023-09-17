#!/bin/bash

# Check if a directory path is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Directory containing CSV files
directory_path="$1"

# Check if the provided path is a directory
if [ ! -d "$directory_path" ]; then
    echo "$directory_path is not a directory."
    exit 1
fi

# Loop through each CSV file in the directory
for file in "$directory_path"/*.csv; do
    echo "Processing $file:"
    # Use awk to extract the last column (assuming columns are comma-separated)
    awk -F ',' '{print $NF}' "$file" | sort | uniq -c
    echo "-----------------------"
done

