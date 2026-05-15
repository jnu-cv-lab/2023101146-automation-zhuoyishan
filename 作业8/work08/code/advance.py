import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")


# 基础CNN模型
class BasicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 修改后的CNN模型
class ModifiedCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(ModifiedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 加载MNIST数据
def load_mnist_data(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    full_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_size = 50000
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# 训练函数
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


# 进阶任务1:比较模型结构
print("\n进阶任务1:比较不同网络结构\n")

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

train_loader, val_loader, test_loader = load_mnist_data(batch_size=BATCH_SIZE)

criterion = nn.CrossEntropyLoss()

# 训练基础模型
print("训练基础CNN模型\n")
basic_model = BasicCNN().to(device)
basic_optimizer = optim.Adam(basic_model.parameters(), lr=LEARNING_RATE)

basic_history = {"train_acc": [], "val_acc": []}
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_epoch(
        basic_model, train_loader, criterion, basic_optimizer
    )
    val_loss, val_acc = evaluate(basic_model, val_loader, criterion)
    basic_history["train_acc"].append(train_acc)
    basic_history["val_acc"].append(val_acc)
    print(
        f"Epoch {epoch:2d}/{NUM_EPOCHS} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
    )

test_loss, test_acc = evaluate(basic_model, test_loader, criterion)
print(f"\n基础模型测试准确率: {test_acc:.2f}%")

# 训练修改后的模型
print("\n训练修改后的CNN模型 (增加卷积层和全连接层)\n")
modified_model = ModifiedCNN().to(device)
modified_optimizer = optim.Adam(modified_model.parameters(), lr=LEARNING_RATE)

modified_history = {"train_acc": [], "val_acc": []}
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_epoch(
        modified_model, train_loader, criterion, modified_optimizer
    )
    val_loss, val_acc = evaluate(modified_model, val_loader, criterion)
    modified_history["train_acc"].append(train_acc)
    modified_history["val_acc"].append(val_acc)
    print(
        f"Epoch {epoch:2d}/{NUM_EPOCHS} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
    )

test_loss, test_acc_modified = evaluate(modified_model, test_loader, criterion)
print(f"\n修改后模型测试准确率: {test_acc_modified:.2f}%")

# 绘制比较图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), basic_history["val_acc"], "b-", label="Basic CNN")
plt.plot(
    range(1, NUM_EPOCHS + 1), modified_history["val_acc"], "r-", label="Modified CNN"
)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Model Structure Comparison")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
accuracies = [test_acc, test_acc_modified]
labels = ["Basic CNN", "Modified CNN"]
colors = ["blue", "red"]
plt.bar(labels, accuracies, color=colors)
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Comparison")
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha="center")

plt.tight_layout()
plt.savefig(
    "/home/shi_chou_chu_jin/cv-course/work09/Advanced_photo/model_comparison.png",
    dpi=150,
)
plt.show()

# 进阶任务2:比较优化器
print("进阶任务2:比较不同优化器")

NUM_EPOCHS_OPT = 10

# 使用基础模型比较优化器
print("\n使用 SGD 优化器训练\n")
model_sgd = BasicCNN().to(device)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
sgd_history = {"train_acc": [], "val_acc": []}
for epoch in range(1, NUM_EPOCHS_OPT + 1):
    train_loss, train_acc = train_epoch(
        model_sgd, train_loader, criterion, optimizer_sgd
    )
    val_loss, val_acc = evaluate(model_sgd, val_loader, criterion)
    sgd_history["train_acc"].append(train_acc)
    sgd_history["val_acc"].append(val_acc)
    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS_OPT} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )
test_loss, test_acc_sgd = evaluate(model_sgd, test_loader, criterion)
print(f"SGD 测试准确率: {test_acc_sgd:.2f}%")

print("\n使用 Adam 优化器训练\n")
model_adam = BasicCNN().to(device)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
adam_history = {"train_acc": [], "val_acc": []}
for epoch in range(1, NUM_EPOCHS_OPT + 1):
    train_loss, train_acc = train_epoch(
        model_adam, train_loader, criterion, optimizer_adam
    )
    val_loss, val_acc = evaluate(model_adam, val_loader, criterion)
    adam_history["train_acc"].append(train_acc)
    adam_history["val_acc"].append(val_acc)
    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS_OPT} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )
test_loss, test_acc_adam = evaluate(model_adam, test_loader, criterion)
print(f"Adam 测试准确率: {test_acc_adam:.2f}%")

# 绘制优化器比较图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS_OPT + 1), sgd_history["val_acc"], "b-", label="SGD")
plt.plot(range(1, NUM_EPOCHS_OPT + 1), adam_history["val_acc"], "r-", label="Adam")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Optimizer Comparison")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
accuracies = [test_acc_sgd, test_acc_adam]
labels = ["SGD", "Adam"]
colors = ["blue", "red"]
plt.bar(labels, accuracies, color=colors)
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy by Optimizer")
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha="center")

plt.tight_layout()
plt.savefig(
    "/home/shi_chou_chu_jin/cv-course/work09/Advanced_photo/optimizer_comparison.png",
    dpi=150,
)
plt.show()

# 优化器比较表格
print("\n" + "-" * 50)
print("优化器比较记录表")
print("-" * 50)
print(f"{'Optimizer':<15} {'Learning Rate':<15} {'Test Accuracy':<15}")
print("-" * 50)
print(f"{'SGD':<15} {'0.01':<15} {test_acc_sgd:.2f}%")
print(f"{'Adam':<15} {'0.001':<15} {test_acc_adam:.2f}%")
print("-" * 50)

# 进阶任务3:CIFAR-10实验
print("\n进阶任务3:CIFAR-10图像分类\n")

# 加载CIFAR-10数据
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

# 划分验证集
train_size = 45000
val_size = 5000
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader_cifar = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader_cifar = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader_cifar = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"CIFAR-10训练集大小: {len(train_dataset)}")
print(f"CIFAR-10验证集大小: {len(val_dataset)}")
print(f"CIFAR-10测试集大小: {len(test_dataset)}")

# CIFAR-10的类别
cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# 显示样本图像
print("\n显示CIFAR-10样本图像\n")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    image, label = test_dataset[i]
    # 反归一化
    image = image * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor(
        [0.4914, 0.4822, 0.4465]
    ).view(3, 1, 1)
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)
    ax = axes[i // 4, i % 4]
    ax.imshow(image)
    ax.set_title(f"Label: {cifar10_classes[label]}", fontsize=10)
    ax.axis("off")
plt.suptitle("CIFAR-10 Dataset Sample Images", fontsize=14)
plt.tight_layout()
plt.savefig(
    "/home/shi_chou_chu_jin/cv-course/work09/Advanced_photo/cifar10_samples.png",
    dpi=150,
)
plt.show()

# 训练CIFAR-10模型
print("\n训练CIFAR-10 CNN模型\n")


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


cifar_model = CIFAR10CNN().to(device)
cifar_optimizer = optim.Adam(cifar_model.parameters(), lr=0.001)
cifar_criterion = nn.CrossEntropyLoss()

NUM_EPOCHS_CIFAR = 10
cifar_history = {"train_acc": [], "val_acc": []}

print("\n训练CIFAR-10 (10 epochs)\n")
for epoch in range(1, NUM_EPOCHS_CIFAR + 1):
    train_loss, train_acc = train_epoch(
        cifar_model, train_loader_cifar, cifar_criterion, cifar_optimizer
    )
    val_loss, val_acc = evaluate(cifar_model, val_loader_cifar, cifar_criterion)
    cifar_history["train_acc"].append(train_acc)
    cifar_history["val_acc"].append(val_acc)
    print(
        f"Epoch {epoch:2d}/{NUM_EPOCHS_CIFAR} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
    )

test_loss, test_acc_cifar = evaluate(cifar_model, test_loader_cifar, cifar_criterion)
print(f"\nCIFAR-10测试准确率: {test_acc_cifar:.2f}%")

# 绘制CIFAR-10训练曲线
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, NUM_EPOCHS_CIFAR + 1),
    cifar_history["train_acc"],
    "b-",
    label="Training Accuracy",
)
plt.plot(
    range(1, NUM_EPOCHS_CIFAR + 1),
    cifar_history["val_acc"],
    "r-",
    label="Validation Accuracy",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CIFAR-10 Training Progress")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(
    "/home/shi_chou_chu_jin/cv-course/work09/Advanced_photo/cifar10_training.png",
    dpi=150,
)
plt.show()

# MNIST vs CIFAR-10 比较
print("\nMNIST vs CIFAR-10 比较\n")

print("\n" + "-" * 50)
print("MNIST 与 CIFAR-10 比较记录表")
print("-" * 50)
print(f"{'数据集':<15} {'图像类型':<15} {'类别数':<10} {'测试准确率':<15} {'难度':<10}")
print("-" * 50)
print(f"{'MNIST':<15} {'灰度手写数字':<15} {'10':<10} {test_acc:.2f}% {'简单':<10}")
print(
    f"{'CIFAR-10':<15} {'彩色自然图像':<15} {'10':<10} {test_acc_cifar:.2f}% {'困难':<10}"
)
print("-" * 50)

print("\n实验结果总结\n")
print(f"1. 基础MNIST CNN测试准确率: {test_acc:.2f}%")
print(f"2. 修改后MNIST CNN测试准确率: {test_acc_modified:.2f}%")
print(f"3. SGD优化器测试准确率: {test_acc_sgd:.2f}%")
print(f"4. Adam优化器测试准确率: {test_acc_adam:.2f}%")
print(f"5. CIFAR-10测试准确率: {test_acc_cifar:.2f}%")
