#!/bin/bash

total_packets=0
tcp_packets=0
udp_packets=0
icmp_packets=0

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
    echo "File: $file"
    
    # Use tshark to count packets and filter by protocol
    packet_stats=$(tshark -r "$file" -qz io,phs)

    # Extract and accumulate the total packet count
    total=$(echo "$packet_stats" | grep "frames:" | awk -F ":" '{sum+=$2} END {print sum}')
    total_packets=$((total_packets + total))

    # Extract and accumulate protocol-specific packet counts (TCP, UDP, ICMP)
    tcp=$(echo "$packet_stats" | grep "tcp.segments" | awk -F ":" '{sum+=$2} END {print sum}')
    tcp_packets=$((tcp_packets + tcp))

    udp=$(echo "$packet_stats" | grep "udp" | awk -F ":" '{sum+=$2} END {print sum}')
    udp_packets=$((udp_packets + udp))

    icmp=$(echo "$packet_stats" | grep "icmp" | awk -F ":" '{sum+=$2} END {print sum}')
    icmp_packets=$((icmp_packets + icmp))
    
    # Display packet counts for this file
    echo "Total Packets: $total"
    echo "TCP Packets: $tcp"
    echo "UDP Packets: $udp"
    echo "ICMP Packets: $icmp"
    
    echo  # Add a newline for better readability
done

# Display the total counts across all files
echo "Total Packets Across All Files: $total_packets"
echo "Total TCP Packets Across All Files: $tcp_packets"
echo "Total UDP Packets Across All Files: $udp_packets"
echo "Total ICMP Packets Across All Files: $icmp_packets"
