#!/bin/bash

# Check difference between the two versions of the labelled CIC-IDS2017 dataset
diff <(head -n1 Downloads/MachineLearningCVE_concatenated_result.csv | tr ',' '\n') <(head -n1 Downloads/TrafficLabelling_concatenated_result.csv| tr ',' '\n')

: <<'COMMENT'
Result:
0a1,4
> Flow ID
>  Source IP
>  Source Port
>  Destination IP
1a6,7
>  Protocol
>  Timestamp
COMMENT