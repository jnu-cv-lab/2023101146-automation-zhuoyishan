import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

BATCH_SIZE = 64
NUM_EPOCHS = 5


# CNN模型
class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        # 卷积层块
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积块1:保留输出用于特征图可视化
        x1 = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x1)
        # 卷积块2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x, x1  # 返回预测结果 + 第一层特征图


# 数据集
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
full_train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
train_size = 50000
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 任务1:训练模型
print("\n任务1:复用模型重新训练")
final_model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    # 训练
    final_model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = final_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / train_total

    # 验证
    final_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = final_model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1} | Train Loss:{train_loss:.3f} | Val Acc:{val_acc:.2f}%")

# 测试模型
final_model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, _ = final_model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
test_loss /= len(test_loader)
test_acc = 100 * test_correct / test_total
print(f"\n最终模型测试准确率: {test_acc:.2f}%")

# 任务2:优化器对比
print("\n任务2:优化器对比")
optim_configs = [
    ("SGD", optim.SGD, {"lr": 0.01}),
    ("SGD+Momentum", optim.SGD, {"lr": 0.01, "momentum": 0.9}),
    ("Adam", optim.Adam, {"lr": 0.001}),
]
optim_results = {}

for name, opt_cls, kwargs in optim_configs:
    model = CNNModel().to(device)
    opt = opt_cls(model.parameters(), **kwargs)
    train_loss_list = []
    val_acc_list = []
    print(f"\n{name} 训练\n")
    for epoch in range(NUM_EPOCHS):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for imgs, lbs in train_loader:
            imgs, lbs = imgs.to(device), lbs.to(device)
            opt.zero_grad()
            out, _ = model(imgs)
            loss = criterion(out, lbs)
            loss.backward()
            opt.step()
            t_loss += loss.item()
            _, pred = out.max(1)
            t_total += lbs.size(0)
            t_correct += pred.eq(lbs).sum().item()
        t_loss /= len(train_loader)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, lbs in val_loader:
                imgs, lbs = imgs.to(device), lbs.to(device)
                out, _ = model(imgs)
                _, pred = out.max(1)
                v_total += lbs.size(0)
                v_correct += pred.eq(lbs).sum().item()
        v_acc = 100 * v_correct / v_total

        train_loss_list.append(t_loss)
        val_acc_list.append(v_acc)
        print(f"Epoch {epoch+1} | Loss:{t_loss:.3f} | Val Acc:{v_acc:.2f}%")

    model.eval()
    te_correct, te_total = 0, 0
    with torch.no_grad():
        for imgs, lbs in test_loader:
            imgs, lbs = imgs.to(device), lbs.to(device)
            out, _ = model(imgs)
            _, pred = out.max(1)
            te_total += lbs.size(0)
            te_correct += pred.eq(lbs).sum().item()
    te_acc = 100 * te_correct / te_total
    optim_results[name] = {
        "loss": train_loss_list,
        "acc": val_acc_list,
        "test_acc": te_acc,
    }
    print(f"{name} 测试准确率: {te_acc:.2f}%")

plt.figure(figsize=(14, 5))
plt.subplot(121)
for k, v in optim_results.items():
    plt.plot(v["loss"], label=k)
    ##优化器损失对比
plt.xlabel("Epoch"), plt.ylabel("Train Loss"), plt.title(
    "Optimizer Loss Comparison"
), plt.legend()
plt.subplot(122)
for k, v in optim_results.items():
    plt.plot(v["acc"], label=k)
    ##优化器准确率对比
plt.xlabel("Epoch"), plt.ylabel("Val Acc"), plt.title(
    "Optimizer Accuracy Comparison"
), plt.legend()
plt.tight_layout()
plt.savefig("/home/shi_chou_chu_jin/cv-course/work09/photo/optimizer_compare.png")
plt.show()

# 任务3:学习率对比
print("\n任务3:学习率对比")
lr_list = [0.1, 0.01, 0.001]
lr_results = {}
for lr in lr_list:
    model = CNNModel().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    acc_list = []
    print(f"\n学习率 {lr} 训练\n")
    for epoch in range(NUM_EPOCHS):
        model.train()
        t_loss = 0
        for imgs, lbs in train_loader:
            imgs, lbs = imgs.to(device), lbs.to(device)
            opt.zero_grad()
            out, _ = model(imgs)
            loss = criterion(out, lbs)
            loss.backward()
            opt.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, lbs in val_loader:
                imgs, lbs = imgs.to(device), lbs.to(device)
                out, _ = model(imgs)
                _, pred = out.max(1)
                v_total += lbs.size(0)
                v_correct += pred.eq(lbs).sum().item()
        v_acc = 100 * v_correct / v_total

        loss_list.append(t_loss)
        acc_list.append(v_acc)
        print(f"Epoch {epoch+1} | Loss:{t_loss:.3f} | Val Acc:{v_acc:.2f}%")

    model.eval()
    te_correct, te_total = 0, 0
    with torch.no_grad():
        for imgs, lbs in test_loader:
            imgs, lbs = imgs.to(device), lbs.to(device)
            out, _ = model(imgs)
            _, pred = out.max(1)
            te_total += lbs.size(0)
            te_correct += pred.eq(lbs).sum().item()
    te_acc = 100 * te_correct / te_total
    lr_results[f"lr={lr}"] = {"loss": loss_list, "acc": acc_list, "test_acc": te_acc}

plt.figure(figsize=(14, 5))
plt.subplot(121)
for k, v in lr_results.items():
    plt.plot(v["loss"], label=k)
    ##学习率损失对比
plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title(
    "Learning Rate Loss Comparison"
), plt.legend()
plt.subplot(122)
for k, v in lr_results.items():
    plt.plot(v["acc"], label=k)
    ##学习率准确率对比
plt.xlabel("Epoch"), plt.ylabel("Acc"), plt.title(
    "Learning Rate Accuracy Comparison"
), plt.legend()
plt.tight_layout()
plt.savefig("/home/shi_chou_chu_jin/cv-course/work09/photo/lr_compare.png")
plt.show()

# 任务4:卷积核可视化
kernels = final_model.conv1.weight.data.cpu().numpy()
plt.figure(figsize=(12, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(kernels[i, 0], cmap="gray")
    plt.title(f"Kernel {i+1}")
    plt.axis("off")
    ##第一层卷积核
plt.suptitle("First Layer Convolution Kernels")
plt.tight_layout()
plt.savefig("/home/shi_chou_chu_jin/cv-course/work09/photo/conv_kernel.png")
plt.show()

# 任务5:特征图可视化
img, label = test_dataset[0]
img_tensor = img.unsqueeze(0).to(device)
final_model.eval()
with torch.no_grad():
    _, feat_map = final_model(img_tensor)
feat_map = feat_map.squeeze(0).cpu().numpy()

plt.figure(figsize=(14, 6))
plt.subplot(3, 3, 1)
plt.imshow(img.squeeze(), cmap="gray")
##原始图像|标签
plt.title(f"Original Image | Label: {label}")
plt.axis("off")
for i in range(8):
    plt.subplot(3, 3, i + 2)
    plt.imshow(feat_map[i], cmap="gray")
    plt.title(f"Feature {i+1}")
    plt.axis("off")
    ##第一层卷积特征图
plt.suptitle("First Layer Convolution Feature Maps")
plt.tight_layout()
plt.savefig("/home/shi_chou_chu_jin/cv-course/work09/photo/feature_map.png")
plt.show()

# 任务6:错误分类样本展示
errors = []
final_model.eval()
with torch.no_grad():
    for idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        out, _ = final_model(x)
        pred = out.argmax(1)
        for i in range(x.size(0)):
            if pred[i] != y[i]:
                errors.append((idx * BATCH_SIZE + i, y[i].item(), pred[i].item()))
                if len(errors) >= 8:
                    break
        if len(errors) >= 8:
            break

plt.figure(figsize=(14, 6))
for i, (img_idx, true, pred) in enumerate(errors[:8]):
    img, _ = test_dataset[img_idx]
    plt.subplot(2, 4, i + 1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"True:{true}\nPred:{pred}", color="red")
    plt.axis("off")
    ##错误分类样本
plt.suptitle("Misclassified Samples")
plt.tight_layout()
plt.savefig("/home/shi_chou_chu_jin/cv-course/work09/photo/error_samples.png")
plt.show()

# 任务7:混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
##测试集混淆矩阵
plt.xlabel("Predicted Label"), plt.ylabel("True Label"), plt.title(
    "Test Set Confusion Matrix"
)
plt.tight_layout()
plt.savefig("/home/shi_chou_chu_jin/cv-course/work09/photo/confusion_matrix.png")
plt.show()
