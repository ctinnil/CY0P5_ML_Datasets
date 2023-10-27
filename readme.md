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
 - [Kaggle version](https://www.kaggle.com/datasets/hassan06/nslkdd)

### 4. **CTU-13**:
 The CTU-13 dataset is particularly notable for its comprehensive representation of botnet traffic, allowing researchers to analyze and develop detection methods for botnet-related activities. CTU-13 features traffic data related to a total of 7 distinct botnets, featuring a mix of both real-world botnet and user-simulated traffic in a university-like environment.
 - [CTU-13 Dataset](https://www.stratosphereips.org/datasets-ctu13)
 - [Kaggle version](https://www.kaggle.com/datasets/dhoogla/ctu13)

### 5. **ISCXIDS2012**:
 The ISCXIDS2012 dataset consists of network traffic data, including both normal network traffic and various types of simulated and real-world cyberattacks.
 - [ISCXIDS2012 Dataset](https://www.unb.ca/cic/datasets/ids.html)
 - [Shortcut to downloads](http://205.174.165.80/CICDataset/ISCX-IDS-2012/Dataset/)

### 6. **CIC-IDS2017**:
 The Canadian Institute for Cybersecurity Intrusion Detection Systems (CICIDS2017) dataset contains network traffic data specific to machine learning for intrusion detection system (IDS) research, describing various attack scenarios, such as DoS, DDoS, and port scanning.
 - [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
 - [Shortcut to downloads](http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/)
 - [Kaggle version](https://www.kaggle.com/datasets/cicdataset/cicids2017)

 ### 7. **CSE-CIC-IDS2018**:
 The colaboorative project between the Communications Security Establishment (CSE) and the Canadian Institute for Cybersecurity (CIC) resulted in a comprehensive dataset, describing various attack scenarios, such as DoS, DDoS, and port scanning, that can be used for machine learning intrusion detection system (IDS) research.
 - [CSE-CIC-IDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)
 - [AWs S3](https://aws.amazon.com/cli/)
 
 ```AWS
 aws s3 sync s3://cse-cic-ids2018/dir/ ./localdir
 ```
 
 - [Kaggle version](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv)

  ### 8. **CIDDS-001 & CIDDS-002**:
 The colaboorative project between the Communications Security Establishment (CSE) and the Canadian Institute for Cybersecurity (CIC) resulted in a comprehensive dataset, describing various attack scenarios, such as DoS, DDoS, and port scanning, that can be used for machine learning intrusion detection system (IDS) research.
 - [CIDDS - COBURG INTRUSION DETECTION DATA SETS](https://www.hs-coburg.de/forschung/forschungsprojekte-oeffentlich/informationstechnologie/cidds-coburg-intrusion-detection-data-sets.html)
 - [Github](https://github.com/markusring/CIDDS)
 - [Shortcut to download CIDDS-001](https://www.hs-coburg.de/fileadmin/hscoburg/WISENT-CIDDS-001.zip)
 - [Shortcut to download CIDDS-002](https://www.hs-coburg.de/fileadmin/hscoburg/WISENT-CIDDS-002.zip)
 - [CIDDS-001 Kaggle version](https://www.kaggle.com/datasets/dhoogla/cidds001)
 - [CIDDS-002 Kaggle version](https://www.kaggle.com/datasets/dhoogla/cidds002)
 
 ### 9. **Kyoto2026+**:
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


# Intrusion Detection System (IDS) Public Datasets Benchmarking

In cybersecurity, the design, development, and implementation of effective Intrusion Detection Systems (IDS) are important for safeguarding IT&C infrastructures from unauthorized access, data breaches, and various forms of malicious activities. The selection of an appropriate ML/DL algorithm plays a essential role in ensuring the security and integrity of protected systems.

But before we can dive in the development of a new-edge algorithm, we shoud have the appropriate data, that needs to be studied and analysed in order to undestant the reality and challenges of our ML problem. In accordance with this paradigm, we chosed to study the early created datasets designed for IDS systems in order to derive leasons learn for feature dataset development.

This experiment aims to comprehensively evaluate the performance of different ML and DL algorithms on a variety of datasets, encompassing a wide range of network traffic scenarios. The datasets used for this analysis include well-known benchmark datasets such as KDD, NSL-KDD, CTU-13, ISCXIDS2012, CIC-IDS2017, CSE-CIC-IDS2018, CIDDS-001/CIDDS-002, and Kyoto 2006+. Each dataset represents a distinct set of challenges and characteristics, making this evaluation both diverse and insightful.

The experiment is divided into three main phases:

1. **Data Acquisition and Preprocessing**:
 - In this phase, we acquire the selected datasets from reputable sources, ensuring the integrity and accuracy of the data.
 - Data preprocessing tasks include handling missing values, selecting the most relevant features using feature selection techniques, normalizing the data, and, if necessary, performing feature engineering to enhance the dataset's suitability for machine learning.

2. **Algorithm Evaluation**:
 - We evaluate the performance of a range of ML/DL algorithms on each dataset. The chosen algorithms include baseline methods like ZeroRule and OneRule, traditional machine learning approaches like Naive Bayes and Random Forest, as well as some of the most used anomaly detection deep learning algorithms.
 - Cross-validation is applied to ensure the robustness of our results. Performance metrics such as precision, variance, and Mean Absolute Error (MAE) are calculated for each algorithm and dataset.

3. **Results and Insights**:
 - The results of this evaluation provide valuable insights into the strengths and weaknesses of different IDS algorithms under various conditions.
 - We analyze the performance of algorithms on both the original datasets and balanced datasets to address the challenge of class imbalance in intrusion detection.
 - Observations and additional details regarding the algorithms' performance are documented, providing a comprehensive overview of their behavior.

By conducting this experiment, we aim to contribute to the understanding of cyber domain dataset generation. The findings will assist in making informed decisions when developing a cybersecurity AI application, by deriving necesary steps and procedures in selecting the appropriate learning data.

The following Jupyter notebooks will provide a detailed walkthrough of the experiments, including code snippets, visualizations, and discussions of the results:
1. [KDD99-BM](benchmarking_IDS_datasets/1.1-2.1_Benchmarking_existing_IDS_datasets_KDD99_v3.0.ipynb)
2. [NSL-KDD-BM](benchmarking_IDS_datasets/1.2-2.2_Benchmarking_existing_IDS_datasets_NSL-KDD_v1.0.ipynb)
3. [CTU-13-BM](benchmarking_IDS_datasets/1.3-2.3_Benchmarking_existing_IDS_datasets_CTU_13_v2_0.ipynb)
4. [ISCXIDS2012-BM](benchmarking_IDS_datasets/1.4-2.4_Benchmarking_existing_IDS_datasets_ISCXIDS2012_v2_0.ipynb)
5. [CIC-IDS2017-BM](benchmarking_IDS_datasets/1.5-2.5_Benchmarking_existing_IDS_datasets_CICIDS2017_v1_0.ipynb)
6. [CSE-CIC-IDS2018-BM](benchmarking_IDS_datasets/1.6-2.6_Benchmarking_existing_IDS_datasets_CSE_CIC_IDS2018_v2_0.ipynb)
7. [CIDDS-001-BM](benchmarking_IDS_datasets/1.7-2.7_Benchmarking_existing_IDS_datasets_CIDDS_001_v2_0.ipynb)
8. [CIDDS-002-BM](benchmarking_IDS_datasets/1.8-2.8_Benchmarking_existing_IDS_datasets_CIDDS_002_v1_0.ipynb)
9. [Kyoto2026+-BM](benchmarking_IDS_datasets/1.9-2.9_Benchmarking_existing_IDS_datasets_Kyoto2026+_v1.0.ipynb)

The results are also saved under the pickle files mentioned below:
1. [KDD99-BM](benchmarking_results/kdd_results.pkl)
2. [NSL-KDD-BM](benchmarking_results/nsl_kdd_results.pkl)
3. [CTU-13-BM](benchmarking_results/ctu13_results.pkl)
4. [ISCXIDS2012-BM](benchmarking_results/iscxids2012_results.pkl)
5. [CIC-IDS2017-BM](bbenchmarking_results/cicids2017_results.pkl)
6. [CSE-CIC-IDS2018-BM](benchmarking_results/csecicids2018_results.pkl)
7. [CIDDS-001-BM](benchmarking_results/cidds001_results.pkl)
8. [CIDDS-002-BM](benchmarking_results/cidds002_results.pkl)
9. [Kyoto2026+-BM](benchmarking_results/kyoto_results.pkl)