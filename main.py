from data import load_data
from svm import run_svm
from qsvm import run_qsvm
from sklearn.metrics import accuracy_score
import numpy as np

from visualize import (
    plot_data,
    plot_result,
    plot_cm,
    plot_roc
)

# =========================
# 1. 数据
# =========================
X_train, X_test, y_train, y_test = load_data()


# =========================
# 2. SVM
# =========================
svm_model = run_svm(X_train, X_test, y_train)
y_svm = svm_model.predict(X_test)

print("\n===== SVM =====")
print("Accuracy:", accuracy_score(y_test, y_svm))

# =========================
# 3. QSVM（量子）
# =========================
qsvm_model, K_train, K_test = run_qsvm(X_train, X_test, y_train)
y_qsvm = qsvm_model.predict(K_test)

print("\n===== QSVM =====")
print("Accuracy:", accuracy_score(y_test, y_qsvm))

# =========================
# 4. 四大图（自动生成）
# =========================

# 数据图
plot_data(X_test, y_test)

# 分类结果
plot_result(X_test, y_svm, "SVM Result")
plot_result(X_test, y_qsvm, "QSVM Result")

# 混淆矩阵
plot_cm(y_test, y_svm, "SVM Confusion Matrix")
plot_cm(y_test, y_qsvm, "QSVM Confusion Matrix")

# ROC
plot_roc(svm_model, X_test, y_test)

from sklearn.metrics import classification_report

print("\n===== SVM Report =====")
print(classification_report(y_test, y_svm))

print("\n===== QSVM Report =====")
print(classification_report(y_test, y_qsvm))