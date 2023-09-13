#!/bin/bash

# Define the string you want to exclude
exclude_string="BENIGN"

# Loop through command-line arguments (assumes you provide the file names as arguments)
for file in "$@"; do
    # Use grep with -c and -v to count lines that do not contain the string
    count=$(grep -c -v "$exclude_string" "$file")
    echo "File: $file, Count: $count"
done
