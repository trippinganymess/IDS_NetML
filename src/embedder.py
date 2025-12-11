import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

DATA_FILE = "/Users/animesh/IDS_NetML/Data/Master.json"
BATCH_SIZE = 256
EMBEDDING_DIM = 128
EPOCHS = 25 
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LIMIT_ROWS = 350000

class NetMLMasterDataset(Dataset):
    """2-pass dataset loader: collects stats, then normalizes."""
    def __init__(self, json_file, limit=100000):
        self.data = []
        self.labels = []
        self.label_map = {} 
        
        self.seq_fields = [
            'intervals_ccnt', 'pld_ccnt', 'hdr_ccnt', 
            'rev_intervals_ccnt', 'rev_pld_ccnt', 'rev_hdr_ccnt',
            'ack_psh_rst_syn_fin_cnt', 'rev_ack_psh_rst_syn_fin_cnt'
        ]
        self.scalar_fields = [
            'pr', 'rev_pld_max', 'rev_pld_mean', 'pld_mean', 
            'pld_median', 'pld_distinct', 'time_length', 
            'bytes_out', 'bytes_in', 'num_pkts_out', 'num_pkts_in',
            'src_port', 'dst_port'
        ]

        # First pass: collect raw scalars for normalization stats
        raw_scalars_list = []
        raw_seq_list = []
        raw_labels = []
        
        print(f"Loading data from {json_file}...")
        with open(json_file, 'r') as f:
            for i, line in tqdm.tqdm(enumerate(f), total=limit, desc="Pass 1: Loading"):
                if i >= limit: break
                try:
                    row = json.loads(line)
                    label_str = row.get('label')
                    if label_str is None: continue 
                    
                    if label_str not in self.label_map:
                        self.label_map[label_str] = len(self.label_map)
                    label_idx = self.label_map[label_str]
                    
                    scalars = []
                    for k in self.scalar_fields:
                        val = float(row.get(k, 0))
                        if k in ['src_port', 'dst_port', 'bytes_in', 'bytes_out', 'num_pkts_out', 'num_pkts_in']:
                            val = np.log1p(val)
                        scalars.append(val)
                    
                    seq_padded = []
                    for k in self.seq_fields:
                        arr = row.get(k, [])
                        if not isinstance(arr, list): arr = [0]
                        if len(arr) < 16:
                            arr = arr + [0]*(16-len(arr))
                        seq_padded.append(arr[:16])
                    
                    raw_scalars_list.append(scalars)
                    raw_seq_list.append(seq_padded)
                    raw_labels.append(label_idx)
                    self.labels.append(label_str)
                except: continue
        
        scalars_arr = np.array(raw_scalars_list, dtype=np.float32)
        self.scalar_mean = scalars_arr.mean(axis=0)
        self.scalar_std = scalars_arr.std(axis=0) + 1e-6
        
        print(f"Pass 2: Normalizing {len(raw_scalars_list)} samples...")
        for i in tqdm.tqdm(range(len(raw_scalars_list)), desc="Pass 2: Normalizing"):
            normed_scalars = (np.array(raw_scalars_list[i]) - self.scalar_mean) / self.scalar_std
            seq_arr = np.array(raw_seq_list[i], dtype=np.float32).T
            
            self.data.append({
                'seq': torch.tensor(seq_arr, dtype=torch.float32),
                'scalar': torch.tensor(normed_scalars, dtype=torch.float32),
                'label': torch.tensor(raw_labels[i], dtype=torch.long)
            })
        
        self.num_classes = len(self.label_map)
        print(f"Loaded {len(self.data)} samples. Found {self.num_classes} unique classes.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- 2. MODEL (ENCODER) with ATTENTION ---
class NetMLEncoder(nn.Module):
    """LSTM + Multi-head Attention encoder with projection head."""
    def __init__(self, num_scalars=13, seq_len=16, num_seq_features=8, embed_dim=32, num_classes=None):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=num_seq_features, hidden_size=128, num_layers=3, batch_first=True, dropout=0.3)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.3,
            batch_first=True
        )
        
        self.scalar_net = nn.Sequential(
            nn.Linear(num_scalars, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        self.scalar_attention = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim) 
        )
        
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x_seq, x_scalar):
        lstm_out, _ = self.lstm(x_seq)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_weights = torch.softmax(attn_out.mean(dim=-1, keepdim=True), dim=1)
        lstm_feat = (attn_out * attn_weights).sum(dim=1)
        
        scalar_feat = self.scalar_net(x_scalar)
        scalar_attn_weights = self.scalar_attention(scalar_feat.unsqueeze(1))
        scalar_feat = scalar_feat * scalar_attn_weights.squeeze(-1)
        
        combined = torch.cat([lstm_feat, scalar_feat], dim=1) 
        embedding = self.projection(combined)
        embedding = F.normalize(embedding, dim=1)
        logits = self.classifier(embedding)
        return embedding, logits

# --- 3. FOCAL LOSS (for handling class imbalance) ---
class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples, focuses on hard negatives."""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """inputs: (batch, num_classes) logits, targets: (batch,) class indices."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == "__main__":
    dataset = NetMLMasterDataset(DATA_FILE, limit=LIMIT_ROWS)
    
    val_split = 0.15
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    num_classes = dataset.num_classes
    print(f"Number of classes: {num_classes}")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    model = NetMLEncoder(num_seq_features=8, embed_dim=EMBEDDING_DIM, num_classes=num_classes).to(DEVICE)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)  # Focal Loss for class imbalance
    print(f"Starting FocalLoss Training (Dim={EMBEDDING_DIM}, Classes={num_classes}) on {DEVICE}...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0
    best_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    os.makedirs('results', exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in pbar:
            seq = batch['seq'].to(DEVICE)
            scalar = batch['scalar'].to(DEVICE)
            label = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            
            embeddings, logits = model(seq, scalar)
            loss = criterion(logits, label)
            _, preds = torch.max(logits, 1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            acc = 100 * correct / total if total > 0 else 0
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{acc:.1f}%"})
        
        train_loss_avg = total_loss / len(train_loader)
        train_acc_avg = (100 * correct / total) if total > 0 else 0
        train_losses.append(train_loss_avg)
        train_accs.append(train_acc_avg)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            pbar_val = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for batch in pbar_val:
                seq = batch['seq'].to(DEVICE)
                scalar = batch['scalar'].to(DEVICE)
                label = batch['label'].to(DEVICE)
                
                embeddings, logits = model(seq, scalar)
                loss = criterion(logits, label)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(label.cpu().numpy())
                
                val_loss += loss.item()
        
        val_loss_avg = val_loss / len(val_loader)
        val_acc_avg = (100 * val_correct / val_total) if val_total > 0 else 0
        val_losses.append(val_loss_avg)
        val_accs.append(val_acc_avg)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.2f}%")
        print(f"  Val Loss:   {val_loss_avg:.4f}, Val Acc:   {val_acc_avg:.2f}%")
        
        scheduler.step(val_loss_avg)
        
        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            best_loss = val_loss_avg
            torch.save(model.state_dict(), "encoder_best.pth")
            print(f"  ✓ Best model saved (Val Acc: {val_acc_avg:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered! Val accuracy has not improved for {patience} epochs.")
                print(f"  Best validation accuracy: {best_val_acc:.2f}%")
                break
    
    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load("encoder_best.pth"))
    
    print("Saving training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(val_losses, label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, label='Train Accuracy', marker='o', markersize=3)
    axes[1].plot(val_accs, label='Val Accuracy', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300)
    print("  ✓ Training curves saved to results/training_curves.png")
    plt.close()
    
    print("Computing validation confusion matrix...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Computing CM"):
            seq = batch['seq'].to(DEVICE)
            scalar = batch['scalar'].to(DEVICE)
            label = batch['label'].to(DEVICE)
            
            embeddings, logits = model(seq, scalar)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    if len(all_preds) > 0:
        cm = confusion_matrix(all_labels, all_preds)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        class_names = sorted(dataset.label_map.keys())
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                           color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=6)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Validation Confusion Matrix')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("  ✓ Confusion matrix saved to results/confusion_matrix.png")
        plt.close()
        
        print("\nValidation Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    print("\nSaving Final Encoder...")
    torch.save(model.state_dict(), "encoder_final.pth")
    
    print("Saving Embeddings...")
    model.eval()
    all_embeds = []
    all_labels = []
    
    extract_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for batch in tqdm.tqdm(extract_loader, desc="Extracting"):
            seq = batch['seq'].to(DEVICE)
            scalar = batch['scalar'].to(DEVICE)
            emb = model(seq, scalar)
            all_embeds.append(emb.cpu().numpy())
            all_labels.append(batch['label'].numpy())
            
    X = np.vstack(all_embeds)
    y = np.concatenate(all_labels)
    np.savez("embeddings.npz", X=X, y=y)
    with open("label_map.json", "w") as f: json.dump(dataset.label_map, f)
        
    print("Done.")
