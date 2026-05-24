import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# ===== 1 数据分布 =====
def plot_data(X, y):
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm")
    plt.title("Dataset Distribution")
    plt.show()

# ===== 2 分类结果 =====
def plot_result(X, y_pred, title):
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y_pred, cmap="coolwarm")
    plt.title(title)
    plt.show()

# ===== 3 混淆矩阵 =====
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title(title)
    plt.show()

# ===== 4 ROC曲线 =====
def plot_roc(svm_model, X_test, y_test):
    RocCurveDisplay.from_estimator(svm_model, X_test, y_test, name="SVM")
    plt.title("SVM ROC Curve")
    plt.show()