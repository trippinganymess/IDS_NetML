import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("LSTM MODEL TRAINING - NetML Cyberattack Detection")
print("="*60)

# Step 1: Load preprocessed data
print("\n[1/5] Loading preprocessed data...")
X_train = np.load('Preprocessing/data/preprocessed/X_train_lstm.npy')
X_val = np.load('Preprocessing/data/preprocessed/X_val_lstm.npy')
X_test = np.load('Preprocessing/data/preprocessed/X_test_lstm.npy')
y_train = np.load('Preprocessing/data/preprocessed/y_train_lstm.npy')
y_val = np.load('Preprocessing/data/preprocessed/y_val_lstm.npy')
y_test = np.load('Preprocessing/data/preprocessed/y_test_lstm.npy')

label_encoder = joblib.load('Preprocessing/data/preprocessed/label_encoder.pkl')

print(f"✓ Train shape: {X_train.shape}")
print(f"✓ Val shape: {X_val.shape}")
print(f"✓ Test shape: {X_test.shape}")
print(f"✓ Classes: {label_encoder.classes_}")

# Step 2: Build LSTM model
print("\n[2/5] Building LSTM model...")

model = Sequential([
    # First LSTM layer
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    
    # Second LSTM layer
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    
    # Dense layers
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Output layer
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model architecture:")
model.summary()

# Step 3: Setup callbacks
print("\n[3/5] Setting up training callbacks...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'models/lstm_best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

import os
os.makedirs('models', exist_ok=True)

# Step 4: Train the model
print("\n[4/5] Training LSTM model...")
print("⏳ This may take 10-15 minutes...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

print("✓ Training complete!")

# Step 5: Evaluate the model
print("\n[5/5] Evaluating model on test set...")

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('LSTM Model - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/lstm_confusion_matrix.png', dpi=300)
print("✓ Confusion matrix saved to results/lstm_confusion_matrix.png")

# Save training history plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/lstm_training_history.png', dpi=300)
print("✓ Training history saved to results/lstm_training_history.png")

# Save final model
model.save('models/lstm_final_model.keras')
print("✓ Final model saved to models/lstm_final_model.keras")

print("\n" + "="*60)
print("✅ LSTM MODEL TRAINING COMPLETE!")
print("="*60)
print(f"📊 Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"📁 Model saved: models/lstm_final_model.keras")
print(f"📁 Results saved: results/")
print("="*60)
