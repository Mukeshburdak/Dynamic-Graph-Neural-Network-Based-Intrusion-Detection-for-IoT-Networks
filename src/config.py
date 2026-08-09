"""
config.py

Central configuration file for the IoT Intrusion Detection project.
Every other module (loader, preprocessing, graph builder, models, training,
evaluation) imports constants from here so that paths and label definitions
stay consistent across the entire pipeline.
"""

import os
import torch

# ---------------------------------------------------------------------------
# 1. PROJECT ROOT AND DIRECTORY PATHS
# ---------------------------------------------------------------------------

# Absolute path to the project root (one level above the "src" folder).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw CICIoT2023 CSVs, organized as one sub-folder per attack (as downloaded
# from the Canadian Institute for Cybersecurity website).
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CICIoT2023")

# Processed, cleaned, and split CSVs (after preprocessing.py runs).
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
VAL_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "validation.csv")
TEST_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")

# Serialized PyTorch Geometric graph objects (after graph_builder.py runs).
GRAPHS_DIR = os.path.join(PROJECT_ROOT, "data", "graphs")

# Saved trained model checkpoints.
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Evaluation outputs (metrics, figures, comparison tables).
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Ensure key output directories exist at import time so downstream modules
# never have to worry about missing folders.
for _directory in [
    PROCESSED_DATA_DIR,
    GRAPHS_DIR,
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
]:
    os.makedirs(_directory, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. LABEL MAPPING (34 individual attack folders -> 7 attack categories)
# ---------------------------------------------------------------------------
# Keys below MUST exactly match the folder names inside data/raw/CICIoT2023/
# as downloaded from the CIC website. Values are the 7 category labels
# (+ "Benign") that the GNN will actually be trained to classify.

ATTACK_FOLDER_TO_CATEGORY = {
    # ---- DDoS ----
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-PSHACK_FLOOD": "DDoS",
    "DDoS-RSTFINFLOOD": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",

    # ---- DoS ----
    "DoS-UDP_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",

    # ---- Mirai (botnet) ----
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",

    # ---- Recon ----
    "Recon-PortScan": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-HostDiscovery": "Recon",
    "Recon-PingSweep": "Recon",
    "VulnerabilityScan": "Recon",

    # ---- Spoofing / MITM ----
    "MITM-ArpSpoofing": "Spoofing",
    "DNS_Spoofing": "Spoofing",

    # ---- Web-based ----
    "XSS": "Web-based",
    "SqlInjection": "Web-based",
    "CommandInjection": "Web-based",
    "BrowserHijacking": "Web-based",
    "Uploading_Attack": "Web-based",
    "Backdoor_Malware": "Web-based",

    # ---- Brute Force ----
    "DictionaryBruteForce": "Brute Force",

    # ---- Benign ----
    "Benign_Final": "Benign",
}

# The 8 final classes the model will output (7 attack categories + Benign).
# Order here defines the integer class index used everywhere downstream.
CLASS_NAMES = [
    "Benign",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-based",
    "Brute Force",
]

# String label -> integer index, and the reverse mapping.
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}

NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# 3. RANDOM SEED (for reproducibility across numpy / torch / sklearn)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 4. DATA SPLIT RATIOS
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ---------------------------------------------------------------------------
# 5. DEVICE CONFIGURATION
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 6. GRAPH CONSTRUCTION SETTINGS
# ---------------------------------------------------------------------------

# Number of flow records grouped together into a single graph snapshot
# (a "window" of traffic used to build one dynamic graph).
GRAPH_WINDOW_SIZE = 1000

# Number of records the window slides forward by each step. A value smaller
# than GRAPH_WINDOW_SIZE creates overlapping windows; equal to it creates
# non-overlapping windows.
GRAPH_WINDOW_STRIDE = 1000

# Maximum number of nodes (flow records) allowed per graph snapshot. If a
# window contains more rows than this, it is randomly subsampled down to
# this size before k-NN edges are computed (keeps graph construction and
# GNN memory usage bounded).
MAX_NODES_PER_GRAPH = 500

# Number of nearest neighbors (by feature-space similarity) each node
# connects to when building a similarity graph. Since CICIoT2023's public
# CSV release does not include Source/Destination IP addresses, graphs are
# built by connecting each flow record to its K most similar flow records
# rather than by real network topology.
K_NEIGHBORS = 8


# ---------------------------------------------------------------------------
# 7. MODEL / TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------

HIDDEN_CHANNELS = 64
NUM_GNN_LAYERS = 2
DROPOUT = 0.3

LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 32
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10


# ---------------------------------------------------------------------------
# 8. FEATURE COLUMNS
# ---------------------------------------------------------------------------
# Confirmed directly from the header row of the raw CICIoT2023 CSV files
# (39 numeric/flag features). This is the standard CICIoT2023 flow-level
# feature schema published by the Canadian Institute for Cybersecurity.
# Order matches the actual CSV column order.

FEATURE_COLUMNS = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]

NUM_FEATURES = len(FEATURE_COLUMNS)

# Name of the raw label column as it appears in the original CICIoT2023 CSVs.
# NOTE: in this folder-per-attack release, the label may or may not be
# present as an explicit column inside each CSV (some releases only encode
# the label via the parent folder name). loader.py checks for this column
# at runtime and falls back to deriving the label from the folder name via
# ATTACK_FOLDER_TO_CATEGORY if the column is missing.
RAW_LABEL_COLUMN = "label"
