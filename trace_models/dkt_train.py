# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python (gpu_env) (Local)
#     language: python
#     name: gpu_env
# ---

# +
import os, csv, random, math, json
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

DATA_TSV = "/home/jupyter/bert/analysis/analysis_asd/DKT/lak_dkt/astra_100schools/sample_dkt_train.tsv"
OUT_DIR  = "/home/jupyter/bert/analysis/analysis_asd/DKT/lak_dkt/astra_100schools/c_out"
K = 2                      # skills: ER=0, ME=1
# D_MODEL = 64
# N_LAYERS = 1
# DROPOUT = 0.1
# LR = 1e-3
# EPOCHS = 20
# BATCH_SIZE = 64
MAX_SEQ_LEN = 200         
SEED = 1337
# for sample data
D_MODEL = 32
N_LAYERS = 1
DROPOUT = 0.2
LR = 5e-3
EPOCHS = 8
BATCH_SIZE = 16

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

#  LOGGING SETUP 
training_logs = []
training_start_time = datetime.now()

def save_training_logs():
    """Save training logs in multiple formats"""
    # Add metadata
    metadata = {
        "model_type": "DKT",
        "dataset": "astra_100schools",
        "config": {
            "K": K,
            "D_MODEL": D_MODEL,
            "N_LAYERS": N_LAYERS,
            "DROPOUT": DROPOUT,
            "LR": LR,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "SEED": SEED
        },
        "total_epochs": len(training_logs),
        "training_started": training_start_time.isoformat(),
        "training_completed": datetime.now().isoformat(),
        "model_path": os.path.join(OUT_DIR, "dkt.pt"),
        "data_path": DATA_TSV
    }
    
    if training_logs:
        metadata["final_metrics"] = training_logs[-1]
        metadata["best_val_auc"] = max(log["val_auc"] for log in training_logs if not math.isnan(log["val_auc"]))
        metadata["best_val_acc"] = max(log["val_acc"] for log in training_logs)
        metadata["final_train_loss"] = training_logs[-1]["train_loss"]
    
    # Save as JSON with metadata
    json_data = {
        "metadata": metadata,
        "training_logs": training_logs
    }
    
    json_path = os.path.join(OUT_DIR, "training_logs.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f" Saved JSON logs to: {json_path}")
    
    # Save as CSV
    csv_path = os.path.join(OUT_DIR, "training_logs.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_auc", "val_acc"])
        writer.writeheader()
        writer.writerows(training_logs)
    print(f" Saved CSV logs to: {csv_path}")
    
    # Save metadata separately
    metadata_path = os.path.join(OUT_DIR, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f" Saved metadata to: {metadata_path}")
    
    # Save summary text file
    summary_path = os.path.join(OUT_DIR, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("DKT Model Training Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {os.path.basename(DATA_TSV)}\n")
        f.write(f"Total Epochs: {len(training_logs)}\n")
        f.write(f"Training started: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Configuration:\n")
        for key, value in metadata["config"].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        if training_logs:
            f.write("Final Metrics:\n")
            f.write(f"  Train Loss: {training_logs[-1]['train_loss']:.4f}\n")
            f.write(f"  Validation AUC: {training_logs[-1]['val_auc']:.4f}\n")
            f.write(f"  Validation Accuracy: {training_logs[-1]['val_acc']:.4f}\n\n")
            
            best_auc = max(log["val_auc"] for log in training_logs if not math.isnan(log["val_auc"]))
            best_acc = max(log["val_acc"] for log in training_logs)
            f.write("Best Metrics:\n")
            f.write(f"  Best Validation AUC: {best_auc:.4f}\n")
            f.write(f"  Best Validation Accuracy: {best_acc:.4f}\n\n")
        
        f.write("Model saved to: " + os.path.join(OUT_DIR, "dkt.pt") + "\n\n")
        
        f.write("Epoch-by-Epoch Results:\n")
        f.write("-" * 50 + "\n")
        for log in training_logs:
            f.write(f"Epoch {log['epoch']:02d} | train_loss {log['train_loss']:.4f} | "
                   f"val_auc {log['val_auc']:.4f} | val_acc {log['val_acc']:.4f}\n")
    
    print(f" Saved summary to: {summary_path}")

# -------------------- DATA --------------------
# Load TSV -> per-student sequences of (skill, correct)
seqs = defaultdict(list)
with open(DATA_TSV) as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        s = row["student_id"]
        k = int(row["skill_id"])
        c = int(row["correct"])
        seqs[s].append((k, c))


# Drop empty and cap length
seqs = {s: v[:MAX_SEQ_LEN] for s, v in seqs.items() if len(v) >= 2}

students = list(seqs.keys())
random.shuffle(students)
split = int(0.8 * len(students))
train_ids, valid_ids = students[:split], students[split:]

print(f"Dataset loaded: {len(students)} students ({len(train_ids)} train, {len(valid_ids)} validation)")

def make_batches(student_ids):
    # produce batches of padded tensors
    batch = []
    for sid in student_ids:
        sc = seqs[sid]
        ks  = np.array([x[0] for x in sc], dtype=np.int64)
        cs  = np.array([x[1] for x in sc], dtype=np.int64)
        batch.append((ks, cs))
    # simple bucketing by length for less padding
    batch.sort(key=lambda x: len(x[0]))
    for i in range(0, len(batch), BATCH_SIZE):
        chunk = batch[i:i+BATCH_SIZE]
        lens = [len(x[0]) for x in chunk]
        T = max(lens)
        B = len(chunk)
        # Inputs X_t are based on (skill_{t-1}, correct_{t-1}); targets are (skill_t, correct_t)
        x = np.zeros((B, T-1, 2*K), dtype=np.float32)
        y_skill = np.full((B, T-1), fill_value=-1, dtype=np.int64)
        y_corr  = np.zeros((B, T-1), dtype=np.float32)
        mask    = np.zeros((B, T-1), dtype=np.float32)
        for b,(ks,cs) in enumerate(chunk):
            L = len(ks)
            if L < 2: 
                continue
            # build t-1 inputs & t targets
            k_prev = ks[:-1]
            c_prev = cs[:-1]
            k_t    = ks[1:]
            c_t    = cs[1:]
            # one-hot 2K: index = k_prev + c_prev*K
            idx = k_prev + c_prev * K
            x[b, :L-1, :] = np.eye(2*K, dtype=np.float32)[idx]
            y_skill[b, :L-1] = k_t
            y_corr[b, :L-1]  = c_t
            mask[b, :L-1]    = 1.0
        yield torch.tensor(x), torch.tensor(y_skill), torch.tensor(y_corr), torch.tensor(mask)

# MODEL 
class DKT(nn.Module):
    def __init__(self, input_dim=2*K, d_model=D_MODEL, n_layers=N_LAYERS, dropout=DROPOUT, K=K):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, d_model, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.head = nn.Linear(d_model, K)  # logits per skill
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B,T,2K]
        h,_ = self.rnn(x)              # [B,T,D]
        logits = self.head(h)          # [B,T,K]
        probs  = self.sigmoid(logits)  # [B,T,K]
        return probs, logits

model = DKT()
opt = torch.optim.Adam(model.parameters(), lr=LR)
bce = nn.BCELoss(reduction='none')

print(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")

# TRAIN / EVAL 
def evaluate(ids):
    model.eval()
    y_true, y_pred, y_acc_true, y_acc_pred = [], [], [], []
    with torch.no_grad():
        for x, y_skill, y_corr, mask in make_batches(ids):
            p, _ = model(x)  # [B,T,K]
            B,T = y_skill.shape
            # gather prob for the actual skill at t
            rows = torch.arange(B).unsqueeze(1).expand(B,T)
            cols = torch.arange(T).unsqueeze(0).expand(B,T)
            sel  = p[rows, cols, y_skill.clamp(min=0)]  # clamp to avoid -1
            m    = (mask * (y_skill!=-1)).float()
            sel  = sel * m
            # collect metrics
            yt = (y_corr * m).flatten().numpy()
            yp = sel.flatten().numpy()
            keep = m.flatten().numpy() > 0
            yt, yp = yt[keep], yp[keep]
            if len(yt):
                y_true.append(yt); y_pred.append(yp)
                y_acc_true.append(yt); y_acc_pred.append((yp>=0.5).astype(np.float32))
    if not y_true:
        return 0.0, 0.0
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    y_acc_true = np.concatenate(y_acc_true); y_acc_pred = np.concatenate(y_acc_pred)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true))>1 else float('nan')
    acc = accuracy_score(y_acc_true, y_acc_pred)
    return auc, acc

#  TRAINING LOOP 
print("Starting training...")
best_auc = -1
for ep in range(1, EPOCHS+1):
    model.train()
    total_loss, total_count = 0.0, 0
    for x, y_skill, y_corr, mask in make_batches(train_ids):
        opt.zero_grad()
        p, _ = model(x)  # [B,T,K]
        B,T = y_skill.shape
        rows = torch.arange(B).unsqueeze(1).expand(B,T)
        cols = torch.arange(T).unsqueeze(0).expand(B,T)
        sel  = p[rows, cols, y_skill.clamp(min=0)]      # [B,T]
        loss_mat = bce(sel, y_corr.float()) * (mask * (y_skill!=-1)).float()
        loss = loss_mat.sum() / (mask.sum() + 1e-8)
        loss.backward()
        opt.step()
        total_loss += loss.item() * float(mask.sum())
        total_count += float(mask.sum())
    train_loss = total_loss / max(total_count,1.0)
    val_auc, val_acc = evaluate(valid_ids)
    
    # Log the epoch results
    epoch_log = {
        "epoch": ep,
        "train_loss": train_loss,
        "val_auc": val_auc,
        "val_acc": val_acc
    }
    training_logs.append(epoch_log)
    
    print(f"Epoch {ep:02d} | train_loss {train_loss:.4f} | val_auc {val_auc:.4f} | val_acc {val_acc:.4f}")
    
    # Update best model if needed
    if not math.isnan(val_auc) and val_auc > best_auc:
        best_auc = val_auc


# Save model
model_path = os.path.join(OUT_DIR, "sample_dkt.pt")
torch.save(model.state_dict(), model_path)
print(f"Saved model to {model_path}")

# Save all training logs
save_training_logs()

print(f"\nTraining completed!")
if training_logs:
    final_metrics = training_logs[-1]
    print(f"Final metrics:")
    print(f"  Train Loss: {final_metrics['train_loss']:.4f}")
    print(f"  Validation AUC: {final_metrics['val_auc']:.4f}")
    print(f"  Validation Accuracy: {final_metrics['val_acc']:.4f}")
    
    best_auc = max(log["val_auc"] for log in training_logs if not math.isnan(log["val_auc"]))
    best_acc = max(log["val_acc"] for log in training_logs)
    print(f"Best metrics:")
    print(f"  Best Validation AUC: {best_auc:.4f}")
    print(f"  Best Validation Accuracy: {best_acc:.4f}")

print(f"\nAll files saved to: {OUT_DIR}")
