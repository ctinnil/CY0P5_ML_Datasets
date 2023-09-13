#!/bin/bash

# Check if a directory path is provided as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/pcap/files"
    exit 1
fi

# Store the directory path provided as an argument
pcap_dir="$1"

# Check if the directory exists
if [ ! -d "$pcap_dir" ]; then
    echo "Directory not found: $pcap_dir"
    exit 1
fi

# Loop through all .pcap files in the specified directory
for file in "$pcap_dir"/*.pcap; do
    if [ -f "$file" ]; then
        echo "File: $file"
        tshark -r "$file" -qz io,phs
    fi
done
