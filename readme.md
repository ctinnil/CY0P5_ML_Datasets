*Before delving into the primary datasets, it's essential to grasp the significance of cybersecurity and why these datasets play a critical role in safeguarding our digital realm. In our interconnected world, cybersecurity threats pose substantial risks to individuals, enterprises, and governments. With the surge in cybercrimes, ranging from data breaches to cyberattacks, having access to trustworthy and current cybersecurity datasets is paramount. These datasets empower us to detect and thwart potential threats effectively.*

# Main datasets used in cybersecurity 

The cybersecurity field is vast, encompassing a wide range of topics and challenges. The community provides a selection of datasets designed for specific research and analysis within the realm of cybersecurity. Here are some key datasets used in cybersecurity, along with brief descriptions and links (as for my study):

### 1. **DARPA Intrusion Detection Data**:
 The DARPA Intrusion Detection Evaluation datasets were collected as part of the 1998 and 1999 DARPA intrusion detection evaluations. These datasets contain a variety of network traffic data for evaluating intrusion detection systems.
 - [1998 DARPA INTRUSION DETECTION EVALUATION DATASET](https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset)
 - [1999 DARPA INTRUSION DETECTION EVALUATION DATASET](https://www.ll.mit.edu/r-d/datasets/1999-darpa-intrusion-detection-evaluation-dataset)
 - [2000 DARPA INTRUSION DETECTION SCENARIO SPECIFIC DATASETS](https://www.ll.mit.edu/r-d/datasets/2000-darpa-intrusion-detection-scenario-specific-datasets)

### 2. **KDD Cup 1999**:
 The KDD Cup 1999 dataset is one of the earliest datasets used for intrusion detection research. It contains data from the DARPA Intrusion Detection Evaluation experiment.
 - [KDD Cup 1999 Dataset](https://www.kdd.org/kdd-cup/view/kdd-cup-1999/Data)
 - [Kaggle version](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)

### 3. **NSL-KDD**:
 The NSL-KDD dataset is a modified version of the well-known KDD Cup 1999 dataset, addressing issues such as redundancy and balance. The new dataset is reduced to the unique values and balanced representation of the different types of the described attacks.
 - [NSL-KDD Dataset](http://www.unb.ca/cic/datasets/nsl.html)
 - [Shortcut to downloads](http://205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip)
 - [Kaggle version](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)

### 4. **CTU-13**:
 The CTU-13 dataset is particularly notable for its comprehensive representation of botnet traffic, allowing researchers to analyze and develop detection methods for botnet-related activities. CTU-13 features traffic data related to a total of 7 distinct botnets, featuring a mix of both real-world botnet and user-simulated traffic in a university-like environment.
 - [CTU-13 Dataset](https://www.stratosphereips.org/datasets-ctu13)

### 5. **ISCXIDS2012**:
 The ISCXIDS2012 dataset consists of network traffic data, including both normal network traffic and various types of simulated and real-world cyberattacks.
 - [ISCXIDS2012 Dataset](https://www.unb.ca/cic/datasets/ids.html)
 - [Shortcut to downloads](http://205.174.165.80/CICDataset/ISCX-IDS-2012/Dataset/)

### 6. **CIC-IDS2017**:
 The Canadian Institute for Cybersecurity Intrusion Detection Systems (CICIDS2017) dataset contains network traffic data specific to machine learning for intrusion detection system (IDS) research, describing various attack scenarios, such as DoS, DDoS, and port scanning.
 - [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
 - [Shortcut to downloads](http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/)

### 7. **Kyoto2026+**:
 The Kyoto (Kyoto 2006+) dataset, also known as the Kyoto University Honeypot Dataset, is a collection of network traffic data specifically designed for cybersecurity research, with a focus on honeypot-based intrusion detection and analysis.
 - [Kyoto 2006+ Dataset](https://datasetsearch.research.google.com/search?ref=TDJjdk1URnFibDg0TTNCNmVRPT0%3D&query=Kyoto%202006%2B&docid=L2cvMTFqbl84M3B6eQ%3D%3D)
 - [Kyoto Data](https://www.takakura.com/Kyoto_data/)
 - [Statistical analysis of honeypot data and building of Kyoto 2006+ dataset for NIDS evaluation](https://dl.acm.org/doi/10.1145/1978672.1978676)

 ### 8. **Hornet**:
 The Hornet datasets consist of a collection of data sets created to explore the potential influence of geographic factors on the occurrence of network attacks. This data was gathered during April and May 2021 from eight identically configured honeypot servers strategically positioned in various regions spanning North America, Europe, and Asia.
 - [Hornet: Network Dataset of Geographically Placed Honeypots](https://www.stratosphereips.org/hornet-network-dataset-of-geographically-placed-honeypots)
 - [Downlaod Hornet 7 Dataset](https://data.mendeley.com/datasets/w6yskg3ffy/3)
 - [Downlaod Hornet 15 Dataset](https://data.mendeley.com/datasets/rry7bhc2f2/2)

## Download 
 To download the files mentioned above, you may access the provided URL directly, or just call the below command.

 ```bash
 wget --mirror -np -nH --cut-dirs=1 -P /path/to/save/directory <URL>
# Example
wget --mirror -np -nH --cut-dirs=1 -P Kyoto2006+ -N -r -A '*' https://www.takakura.com/Kyoto_data/new_data201704/
# `-r``: Recursively download files.
# `-A '*'`: Download files with any extension.
# `-N`: Download only  files that are newer.
 ```


y. **UNSW-NB15**:
 - Description: The UNSW-NB15 dataset is another network intrusion detection dataset containing diverse network traffic data. It includes normal traffic and a wide range of attacks, making it suitable for evaluating intrusion detection systems.
 - Link: [UNSW-NB15 Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)



5. **MAWILab**:
 - Description: The MAWILab dataset consists of real-world traffic data collected from the MAWI (Monitoring and Analysis of Internet Wide-Area Network Traffic) project. It is used for anomaly detection and network traffic analysis.
 - Link: [MAWILab Dataset](http://www.fukuda-lab.org/mawilab/)



7. **Microsoft Malware Classification Challenge (BIG 2015)**:
 - Description: This dataset is used for malware classification tasks. It contains a large collection of files, each labeled as benign or malicious, making it suitable for machine learning-based malware detection.
 - Link: [Microsoft Malware Classification Challenge](https://www.kaggle.com/c/malware-classification/data)

8. **VirusTotal Dataset**:
 - Description: VirusTotal provides a dataset containing various information about files, URLs, and domains. It's useful for research related to malware analysis and threat intelligence.
 - Link: [VirusTotal Public API](https://developers.virustotal.com/reference#getting-started)

*Please note that the availability and specifics of these datasets may change over time, and it's important to review the dataset documentation and terms of use before using them for research or analysis.*


