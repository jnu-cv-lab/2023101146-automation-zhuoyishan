import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import json
from model import SkeletonTransformer

DATA_DIR = "/home/shi_chou_chu_jin/cv-course/work13/data/preprocessed"
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SkeletonTransformer(
    input_dim=132,
    seq_len=30,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    num_classes=6,
    dropout=0.1,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)
        _, pred = out.max(1)
        train_correct += pred.eq(yb).sum().item()
        train_total += yb.size(0)
    train_loss /= train_total
    train_acc = 100.0 * train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * Xb.size(0)
            _, pred = out.max(1)
            val_correct += pred.eq(yb).sum().item()
            val_total += yb.size(0)
    val_loss /= val_total
    val_acc = 100.0 * val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            model.state_dict(),
            "/home/shi_chou_chu_jin/cv-course/work13/data/best_model.pth",
        )

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

print(f"Best val acc: {best_val_acc:.2f}%")

model.load_state_dict(
    torch.load("/home/shi_chou_chu_jin/cv-course/work13/data/best_model.pth")
)
model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(DEVICE)
        out = model(Xb)
        _, pred = out.max(1)
        all_pred.extend(pred.cpu().numpy())
        all_true.extend(yb.numpy())
test_acc = 100.0 * np.sum(np.array(all_pred) == np.array(all_true)) / len(all_true)
print(f"Final test accuracy: {test_acc:.2f}%")

# 混淆矩阵
cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Acc: {test_acc:.2f}%)")
plt.savefig(
    "/home/shi_chou_chu_jin/cv-course/work13/photo/confusion_matrix.png", dpi=150
)
plt.show()

# 训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.tight_layout()
plt.savefig(
    "/home/shi_chou_chu_jin/cv-course/work13/photo/training_curves.png", dpi=150
)
plt.show()
