import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
import joblib
import os

print("="*60)
print("PREPROCESSING PIPELINE - NetML Dataset")
print("="*60)

# Step 1: Load the labeled data
print("\n[1/8] Loading labeled data...")
df = pd.read_pickle('data/labeled_data.pkl')
print(f"✓ Loaded dataset shape: {df.shape}")
print(f"✓ Label distribution:\n{df['label'].value_counts()}")

# Step 2: Separate features and labels
print("\n[2/8] Separating features and labels...")
label_col = 'label'
sequence_cols = ['intervals_seq', 'pld_seq', 'hdr_seq']
exclude_cols = [label_col, 'id'] + sequence_cols
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols].fillna(0)
y = df[label_col]
print(f"✓ Number of features: {len(feature_cols)}")

# Step 3: Encode labels
print("\n[3/8] Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"✓ Classes: {label_encoder.classes_}")
print(f"✓ Encoded distribution: {dict(zip(label_encoder.classes_, np.bincount(y_encoded)))}")

# Step 4: Feature scaling
print("\n[4/8] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Features scaled with StandardScaler")

# Step 5: Class balancing with SMOTE-ENN
print("\n[5/8] Applying SMOTE-ENN for class balancing...")
print("⏳ This may take 2-3 minutes...")
smote_enn = SMOTEENN(random_state=42, n_jobs=-1)
X_balanced, y_balanced = smote_enn.fit_resample(X_scaled, y_encoded)
print(f"✓ After balancing: {X_balanced.shape}")
print(f"✓ Balanced distribution: {dict(zip(label_encoder.classes_, np.bincount(y_balanced)))}")

# Step 6: Prepare sequence features for LSTM
print("\n[6/8] Preparing sequences for LSTM...")
sequences_data = df[sequence_cols].values
sequences_list = []

for i in range(len(sequences_data)):
    combined_seq = np.concatenate([
        sequences_data[i][0],  # intervals_seq (16)
        sequences_data[i][1],  # pld_seq (16)
        sequences_data[i][2]   # hdr_seq (12)
    ])
    sequences_list.append(combined_seq)

sequences_array = np.array(sequences_list)
timesteps = 44  # 16 + 16 + 12
X_lstm_base = sequences_array.reshape(len(sequences_array), timesteps, 1)

# Match balanced data size
min_size = min(len(X_balanced), len(X_lstm_base))
X_lstm = X_lstm_base[:min_size]
X_balanced_final = X_balanced[:min_size]
y_balanced_final = y_balanced[:min_size]
print(f"✓ LSTM sequences prepared: shape {X_lstm.shape}")

# Step 7: Train-validation-test split
print("\n[7/8] Splitting data (70% train, 15% val, 15% test)...")

# XGBoost splits
X_train_xgb, X_temp_xgb, y_train_xgb, y_temp_xgb = train_test_split(
    X_balanced_final, y_balanced_final, test_size=0.3, random_state=42, stratify=y_balanced_final
)
X_val_xgb, X_test_xgb, y_val_xgb, y_test_xgb = train_test_split(
    X_temp_xgb, y_temp_xgb, test_size=0.5, random_state=42, stratify=y_temp_xgb
)

# LSTM splits
X_train_lstm, X_temp_lstm, y_train_lstm, y_temp_lstm = train_test_split(
    X_lstm, y_balanced_final, test_size=0.3, random_state=42, stratify=y_balanced_final
)
X_val_lstm, X_test_lstm, y_val_lstm, y_test_lstm = train_test_split(
    X_temp_lstm, y_temp_lstm, test_size=0.5, random_state=42, stratify=y_temp_lstm
)

print("✓ Data split complete:")
print(f"  XGBoost - Train: {X_train_xgb.shape[0]}, Val: {X_val_xgb.shape[0]}, Test: {X_test_xgb.shape[0]}")
print(f"  LSTM    - Train: {X_train_lstm.shape[0]}, Val: {X_val_lstm.shape[0]}, Test: {X_test_lstm.shape[0]}")

# Step 8: Save all preprocessed data
print("\n[8/8] Saving preprocessed data...")
os.makedirs('data/preprocessed', exist_ok=True)

# Save XGBoost data
np.save('data/preprocessed/X_train_xgb.npy', X_train_xgb)
np.save('data/preprocessed/X_val_xgb.npy', X_val_xgb)
np.save('data/preprocessed/X_test_xgb.npy', X_test_xgb)
np.save('data/preprocessed/y_train_xgb.npy', y_train_xgb)
np.save('data/preprocessed/y_val_xgb.npy', y_val_xgb)
np.save('data/preprocessed/y_test_xgb.npy', y_test_xgb)

# Save LSTM data
np.save('data/preprocessed/X_train_lstm.npy', X_train_lstm)
np.save('data/preprocessed/X_val_lstm.npy', X_val_lstm)
np.save('data/preprocessed/X_test_lstm.npy', X_test_lstm)
np.save('data/preprocessed/y_train_lstm.npy', y_train_lstm)
np.save('data/preprocessed/y_val_lstm.npy', y_val_lstm)
np.save('data/preprocessed/y_test_lstm.npy', y_test_lstm)

# Save preprocessing objects
joblib.dump(scaler, 'data/preprocessed/scaler.pkl')
joblib.dump(label_encoder, 'data/preprocessed/label_encoder.pkl')

print("✓ All data saved to data/preprocessed/")

print("\n" + "="*60)
print("✅ PREPROCESSING COMPLETE!")
print("="*60)
print(f"📊 Features: {X_train_xgb.shape[1]}")
print(f"📊 Classes: {len(label_encoder.classes_)} ({', '.join(label_encoder.classes_)})")
print(f"📊 Training samples: {X_train_xgb.shape[0]}")
print(f"📊 LSTM sequence shape: (samples={X_train_lstm.shape[0]}, timesteps={X_train_lstm.shape[1]}, features={X_train_lstm.shape[2]})")
print("="*60)
