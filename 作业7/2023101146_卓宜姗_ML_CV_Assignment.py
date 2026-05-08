import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

##任务1:数据准备
print("\n任务1:\n")
digits = load_digits()

# 1.数据集中图像的数量
# data形状为(样本数,特征数)，images形状为(样本数,8,8)
n_samples = digits.images.shape[0]
print(f"数据集中图像的总数量:{n_samples}")

# 2.查看每张图像的大小
image_shape = digits.images.shape[1:]
print(f"每张图像的大小:{image_shape[0]}x{image_shape[1]}像素")

# 3.查看类别标签
print(f"类别标签:{digits.target_names}")

# 4.显示若干张样本图像及其真实标签
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f"label:{digits.target[i]}")
    plt.axis("off")
plt.savefig("/home/shi_chou_chu_jin/cv-course/work08/photo/task1_images.png", dpi=300, bbox_inches="tight")
plt.show()

##任务2:数据划分
print("\n任务2:\n")
X = digits.data
y = digits.target
# 1.划分:测试集比例25%(0.25)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"训练集样本数量:{X_train.shape[0]}")
print(f"测试集样本数量:{X_test.shape[0]}")
print("说明:")
print("训练集(TrainingSet):用于拟合模型，让模型学习数据中的规律。")
print("测试集(TestSet):用于评估训练好的模型在未见数据上的泛化能力。")

##任务3:特征表示
print("\n任务3:\n")
# 说明:
# digits.data已经是处理好的特征向量。
# 1.8x8->64维:通过将二维矩阵的行首尾相接，拼接成一个一维数组。
# 2.传统机器学习（如SVM,LR）通常要求输入是固定长度的一维向量，它们不像CNN那样能自动提取空间特征，因此需要手动“扁平化”。
# 3.优点:简单直接，保留了所有像素信息。缺点:丢失了空间位置关系，对旋转、平移敏感，特征维度较高。

print("特征转换说明:")
print(f"原始图像形状:{digits.images[0].shape}")
print(f"转换后特征向量形状:{digits.data[0].shape}")
print("1.一张8x8的图像通过按行(或按列)拼接,变成了长度为64的一维向量。")
print("2.传统机器学习模型输入通常要求是结构化的一维表格数据，无法直接处理二维矩阵。")
print("3.原始像素作为特征的优点是无需复杂特征工程.缺点是忽略了像素间的空间拓扑关系。")

##任务4:模型训练
print("\n任务4:\n")
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),  # 增加迭代次数确保收敛
    "SVM": SVC(gamma="scale"),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
results = {}
for model_name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    results[model_name] = acc
    print(f"{model_name}测试准确率:{acc:.4f}")

##任务5:结果比较
print("\n任务5:\n")

df_results = pd.DataFrame(list(results.items()), columns=["模型", "测试准确率"])
df_results = df_results.sort_values(by="测试准确率", ascending=False)
print("\n各模型测试准确率对比表:")
print(df_results.to_string(index=False))

best_model = df_results.iloc[0]["模型"]
worst_model = df_results.iloc[-1]["模型"]
print(f"\n结果分析:")
print(f"1.准确率最高的模型是:{best_model}")
print(f"2.准确率最低的模型是:{worst_model}")
print(f"3.不同模型之间的表现通常存在明显差异。")
print(f"4.差异原因简析:SVM/KNN/Random Forest 对这种简单像素数据通常表现较好；")
print(f"   朴素贝叶斯假设特征独立，在像素数据上假设较强，表现可能一般；")
print(f"   单棵决策树容易过拟合，表现通常不如集成学习(Random Forest)稳定。")

# 任务6:错误样本分析(以表现较好的SVM为例)
print("\n任务6:\n")

best_clf = SVC(gamma="scale")
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

# 1.绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Task6:Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(
        j,
        i,
        cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.tight_layout()
plt.savefig("/home/shi_chou_chu_jin/cv-course/work08/photo/task6_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# 2.找出被错误分类的样本
errors_idx = np.where(y_pred != y_test)[0]

print(f"测试集中共有{len(errors_idx)}个样本被错误分类。")

if len(errors_idx) > 0:
    plt.figure(figsize=(10, 8))
    # 最多显示8个错误样本
    show_n = min(8, len(errors_idx))

    for i in range(show_n):
        idx = errors_idx[i]
        img = X_test[idx].reshape(8, 8)
        true_label = y_test[idx]
        pred_label = y_pred[idx]

        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title(f"True:{true_label} Predicted:{pred_label}")
        plt.axis("off")

    plt.suptitle("Task6:Misclassified Sample Examples")
    plt.savefig("/home/shi_chou_chu_jin/cv-course/work08/photo/task6_misclassified_samples.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("4.错误原因分析:")
    print(" 观察混淆矩阵和错误样本，通常'1'和'7'、'3'和'8'、'8'和'9'容易混淆。")
    print(" 原因:在8x8的低分辨率下,某些数字的手写体笔画结构非常相似，")
    print("仅依靠原始像素值的模型难以捕捉细微的拓扑差异。")
else:
    print("模型预测全部正确，无可展示的错误样本。")
