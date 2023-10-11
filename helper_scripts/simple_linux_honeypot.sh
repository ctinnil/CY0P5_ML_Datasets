### Configuring iptables rules to log and drop incoming packets
iptables -N LOGGING # Creating a custom iptables chain for logging
iptables -A INPUT -j LOGGING # Logging all incoming packets
iptables -A LOGGING -j LOG --log-prefix "IPTables-Dropped: " --log-level 4 # Dropping all logged packets
iptables -A LOGGING -j DROP # Applying the log and drop rule to the INPUT chain for redirecting, logging, and dropping incoming packets

### Installing and configuring tcpdump to capture the logged packets
tcpdump -i <interface> -w /path/to/logfile.pcap -nn -vvv -s0 -C 3000
#tcpdump -i <interface> -w /path/to/logfile.pcap -nn -vvv -s0 -c <packet_count> port not 22

### Logging SSH connections separately using tcpdump
#tcpdump -i eth0 -nn -p -s 0 -A port 22 or not port 22
