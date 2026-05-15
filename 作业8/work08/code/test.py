import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 设置随机种子,保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 任务1：检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到GPU,使用CPU训练")

# 任务2：加载图像数据集
def load_datasets(batch_size=64):
    # 加载MNIST数据集并划分为训练集、验证集和测试集
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST的均值和标准差
        ]
    )

    # 下载并加载训练集
    full_train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # 加载测试集
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # 将训练集划分为训练集(50000张)和验证集(10000张)
    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 增加返回test_dataset
    return train_loader, val_loader, test_loader, test_dataset


def show_sample_images(dataset, num_images=8):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    # 获取原始MNIST数据集
    original_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )

    for i in range(num_images):
        image, label = original_dataset[i]
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"Label: {label}", fontsize=12)
        axes[i].axis("off")

    plt.suptitle("MNIST Dataset Sample Images", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        "/home/shi_chou_chu_jin/cv-course/work09/photo/sample_images.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


# 任务3：定义CNN模型
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
        # 卷积块1: 输入 28x28x1 -> 28x28x32 -> 池化后 14x14x32
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # 卷积块2: 14x14x32 -> 14x14x64 -> 池化后 7x7x64
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # 展平: 7x7x64 = 3136
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def print_model_summary(model, input_size=(1, 28, 28)):
    print("\n" + "=" * 50)
    print("模型结构:")
    print("=" * 50)
    print(model)
    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("=" * 50 + "\n")


# 任务4、5：训练和验证模型
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    # 验证模型
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n训练:\n")

    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 打印进度
        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    print("\n\n")

    return history


# 任务6：测试模型
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct / total

    print("测试集结果:")
    print(f"测试集 Loss: {test_loss:.4f}")
    print(f"测试集 Accuracy: {test_acc:.2f}%")

    return test_loss, test_acc, all_predictions, all_labels


def show_test_predictions(model, test_dataset, num_images=8, device="cpu"):
    model.eval()

    # 获取原始测试集
    original_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=None
    )

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    with torch.no_grad():
        for i in range(num_images):
            # 获取图像和标签
            image_tensor, true_label = test_dataset[i]
            image_tensor = image_tensor.unsqueeze(0).to(device)  # 添加batch维度

            # 预测
            output = model(image_tensor)
            _, predicted = output.max(1)
            predicted = predicted.item()

            # 显示图像
            original_image, _ = original_testset[i]
            axes[i].imshow(original_image, cmap="gray")
            axes[i].set_title(f"true: {true_label}\npred: {predicted}", fontsize=12)
            color = "green" if true_label == predicted else "red"
            axes[i].title.set_color(color)
            axes[i].axis("off")

    plt.suptitle("Test Set Predictions (Green=Correct, Red=Wrong)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        "/home/shi_chou_chu_jin/cv-course/work09/photo/test_predictions.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


# 任务7：绘制训练曲线
def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # 创建2x1的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss曲线
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss Curves", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy曲线
    ax2.plot(epochs, history["train_acc"], "b-", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Training and Validation Accuracy Curves", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/home/shi_chou_chu_jin/cv-course/work09/photo/training_curves.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


# 任务8：结果分析
def analyze_misclassifications(all_labels, all_predictions):
    from collections import Counter

    misclassified = []
    for true_label, pred_label in zip(all_labels, all_predictions):
        if true_label != pred_label:
            misclassified.append((true_label, pred_label))

    print("\n" + "=" * 50)
    print("错误分类分析:")
    print(f"总测试样本数: {len(all_labels)}")
    print(f"错误分类数量: {len(misclassified)}")
    print(f"错误率: {100 * len(misclassified) / len(all_labels):.2f}%")

    # 统计每个数字的错误情况
    error_by_true = Counter()
    error_pairs = Counter()

    for true_label, pred_label in misclassified:
        error_by_true[true_label] += 1
        error_pairs[(true_label, pred_label)] += 1

    print("\n各数字被错误分类的次数:")
    for digit in range(10):
        print(f"  数字 {digit}: {error_by_true.get(digit, 0)} 次")

    print("\n最常见的错误分类对:")
    for (true_label, pred_label), count in error_pairs.most_common(5):
        print(f"  {true_label} 被误判为 {pred_label}: {count} 次")

    # 生成混淆矩阵
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        "/home/shi_chou_chu_jin/cv-course/work09/photo/confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    print(f"混淆矩阵已保存为 confusion_matrix.png")

    return misclassified


print("PyTorch 图像分类实验 - MNIST手写数字识别")

# 超参数设置
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# 任务2：加载数据
print("\n任务2:加载数据集")
train_loader, val_loader, test_loader, test_dataset = load_datasets(batch_size=BATCH_SIZE)
show_sample_images(None, num_images=8)

# 任务3：定义模型
print("\n任务3:定义CNN模型")
model = CNNModel(num_classes=10).to(device)
print_model_summary(model)

# 任务4：训练设置
print("\n任务4、5:设置训练参数")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"优化器: Adam")
print(f"学习率: {LEARNING_RATE}")
print(f"损失函数: CrossEntropyLoss")
print(f"训练轮数: {NUM_EPOCHS}")
print(f"批次大小: {BATCH_SIZE}")

# 训练和验证
history = train_model(
    model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device
)

# 任务6：测试模型
print("\n任务6:测试模型")
test_loss, test_acc, all_predictions, all_labels = test_model(
    model, test_loader, criterion, device
)

# 显示测试图像预测结果
show_test_predictions(model, test_dataset, num_images=8, device=device)

# 任务7：绘制训练曲线
plot_training_curves(history)

# 任务8：结果分析
print("\n任务8:结果分析:\n")
misclassified = analyze_misclassifications(all_labels, all_predictions)

print("实验总结:\n")
print(f"最终测试准确率: {test_acc:.2f}%")
print(f"最佳验证准确率: {max(history['val_acc']):.2f}%")
print(f"最佳训练准确率: {max(history['train_acc']):.2f}%")
