import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
import joblib
import os

# Loading data
data = []
training_path = 'data/trainingData.json'
if not os.path.exists(training_path):
    print(f"Error: training data file not found at {training_path}.\nPlease place your training JSON lines file at {training_path} or update the script to point to the correct path.")
    # Exit gracefully since downstream processing requires the training data
    raise SystemExit(1)

with open(training_path, 'r') as f:
    for i, line in enumerate(f):
        # limit to ~50% of dataset for quicker runs (preserves original intent)
        if i > 0.5 * 387268:
            break
        if line.strip():
            data.append(json.loads(line))

print(f"Loaded {len(data)} network flow records")

# Step 2: Flatten nested structure
def flatten_flow_record(record):
    flat_record = {}
    scalar_features = ['pr', 'dst_port', 'src_port', 'bytes_in', 'bytes_out',
                      'num_pkts_in', 'num_pkts_out', 'time_length',
                      'pld_mean', 'pld_max', 'pld_median', 'pld_distinct',
                      'rev_pld_mean', 'rev_pld_max', 'rev_pld_distinct',
                      'hdr_mean', 'hdr_distinct', 'rev_hdr_distinct',
                      'pld_bin_inf', 'hdr_bin_40', 'rev_hdr_bin_40',
                      'rev_pld_bin_128', 'rev_pld_var']
    for feat in scalar_features:
        flat_record[feat] = record.get(feat, 0)
    array_features = {
        'intervals_ccnt': 16,
        'rev_intervals_ccnt': 16,
        'pld_ccnt': 16,
        'rev_pld_ccnt': 16,
        'hdr_ccnt': 12,
        'rev_hdr_ccnt': 12,
        'ack_psh_rst_syn_fin_cnt': 5,
        'rev_ack_psh_rst_syn_fin_cnt': 5
    }
    for feat_name, size in array_features.items():
        arr = record.get(feat_name, [0] * size)
        flat_record[f'{feat_name}_sum'] = sum(arr)
        flat_record[f'{feat_name}_mean'] = np.mean(arr)
        flat_record[f'{feat_name}_std'] = np.std(arr)
        flat_record[f'{feat_name}_max'] = max(arr)
        flat_record[f'{feat_name}_min'] = min(arr)
    flat_record['intervals_seq'] = record.get('intervals_ccnt', [0] * 16)
    flat_record['pld_seq'] = record.get('pld_ccnt', [0] * 16)
    flat_record['hdr_seq'] = record.get('hdr_ccnt', [0] * 12)
    flat_record['id'] = record.get('id', 0)
    return flat_record

flattened_data = [flatten_flow_record(rec) for rec in data]
df = pd.DataFrame(flattened_data)

print(f"Flattened dataset shape: {df.shape}")

# Step 3: Assign labels using an annotations mapping (id -> label) if available
annotations_path = 'data/training_annotations.json'
if os.path.exists(annotations_path):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    # annotations keys are strings in the provided file; df['id'] is numeric
    def lookup_label(x):
        # try integer then string key
        try:
            k1 = str(int(x))
        except Exception:
            k1 = str(x)
        return annotations.get(k1, 'unknown')

    df['label'] = df['id'].apply(lookup_label)
    print(f"Assigned labels from {annotations_path}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
else:
    # Fallback: attempt to parse dataset-level labels file if annotations not present
    labels = []
    labels_path = 'data/labels.txt'
    if not os.path.exists(labels_path):
        print(f"Error: neither annotations file ({annotations_path}) nor labels file ({labels_path}) found.\nPlease provide one of them in data/.")
        raise SystemExit(1)

    with open(labels_path, 'r') as f:
        content = f.read().strip()
        if content.startswith('{'):
            try:
                mapping = json.loads(content)
                if 'NetML' in mapping:
                    labels_str = mapping['NetML']
                else:
                    labels_str = next(iter(mapping.values()))
                labels = [s.strip() for s in labels_str.split(';') if s.strip()]
            except json.JSONDecodeError:
                labels = [s.strip() for s in content.split(';') if s.strip()]
        else:
            if ';' in content:
                labels = [s.strip() for s in content.split(';') if s.strip()]
            elif ',' in content:
                labels = [s.strip() for s in content.split(',') if s.strip()]
            else:
                labels = [s.strip() for s in content.splitlines() if s.strip()]

    labels = labels[:len(data)]
    print(f"Loaded {len(labels)} labels from {labels_path}")
    if len(labels) != len(data):
        print("Warning: Number of labels and data samples do not match.")

    # Map detailed labels to broad categories if needed
    label_mapping = {
        'Adload': 'malware', 'Artemis': 'malware', 'BitCoinMiner': 'malware',
        'CCleaner': 'malware', 'Cobalt': 'malware', 'Downware': 'malware',
        'Dridex': 'malware', 'Emotet': 'malware', 'HTBot': 'malware',
        'MagicHound': 'malware', 'MinerTrojan': 'malware', 'PUA': 'malware',
        'Ramnit': 'malware', 'Sality': 'malware', 'Tinba': 'malware',
        'TrickBot': 'malware', 'Trickster': 'malware', 'TrojanDownloader': 'malware',
        'Ursnif': 'malware', 'WebCompanion': 'malware', 'Benign': 'benign'
    }

    mapped_labels = []
    for lbl in labels:
        parts = lbl.split(';')
        detailed_label = parts[-1] if parts else lbl
        mapped_labels.append(label_mapping.get(detailed_label, 'unknown'))

    df['label'] = mapped_labels
    print(f"Label distribution:\n{df['label'].value_counts()}")

# Step 5: Proceed with preprocessing as before - separate features and labels
label_col = 'label'
sequence_cols = ['intervals_seq', 'pld_seq', 'hdr_seq']
exclude_cols = [label_col, 'id', 'sa', 'da'] + sequence_cols
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols].fillna(0)
y = df[label_col]

# Step 6: Encode labels and scale features
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Classes: {label_encoder.classes_}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import joblib
os.makedirs('data/preprocessed', exist_ok=True)
joblib.dump(scaler, 'data/preprocessed/scaler.pkl')
joblib.dump(label_encoder, 'data/preprocessed/label_encoder.pkl')

print("Preprocessing completed with real labels")

df.to_pickle('data/labeled_data.pkl')
print("✓ Labeled data saved to data/labeled_data.pkl")
