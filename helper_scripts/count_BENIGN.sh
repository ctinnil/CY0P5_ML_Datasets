#!/bin/bash

# Define the pattern you want to search for
pattern="BENIGN"

# Loop through command line arguments (assumes you provide the file names as arguments)
for file in "$@"; do
    # Use grep with -c to count matches in each file
    count=$(grep -c "$pattern" "$file")
    echo "File: $file, Count: $count"
done
