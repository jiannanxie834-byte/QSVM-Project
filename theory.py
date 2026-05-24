import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# =========================
# 1. 核方法升维示意图
# =========================
def plot_kernel_mapping():
    np.random.seed(42)

    theta = np.linspace(0, np.pi, 50)
    x1 = np.cos(theta) + np.random.normal(0, 0.1, 50)
    y1 = np.sin(theta) + np.random.normal(0, 0.1, 50)

    x2 = np.cos(theta + np.pi) + np.random.normal(0, 0.1, 50)
    y2 = np.sin(theta + np.pi) + np.random.normal(0, 0.1, 50)

    z1 = x1**2 + y1**2
    z2 = x2**2 + y2**2

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121)
    ax1.scatter(x1, y1, c='blue', label='Class 0')
    ax1.scatter(x2, y2, c='red', label='Class 1')
    ax1.set_title("Original Feature Space")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x1, y1, z1, c='blue')
    ax2.scatter(x2, y2, z2, c='red')
    ax2.set_title("Kernel Feature Space (Higher Dimension)")

    plt.tight_layout()
    plt.savefig("outputs/kernel_mapping.png", dpi=300)
    plt.close()


# =========================
# 2. QSVM流程图
# =========================
def plot_qsvm_flowchart():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    steps = ["Data", "Quantum Encoding", "Kernel Matrix", "Classical Optimization", "Prediction"]
    x = np.linspace(1, 9, len(steps))

    for i, (xi, text) in enumerate(zip(x, steps)):
        ax.text(xi, 3, text, ha='center', bbox=dict(boxstyle="round", fc="lightblue"))

        if i < len(steps) - 1:
            ax.arrow(xi+0.4, 3, 0.8, 0, head_width=0.2)

    plt.title("QSVM Pipeline")
    plt.savefig("outputs/qsvm_flowchart.png", dpi=300)
    plt.close()


# =========================
# 3. 经典 vs 量子结构图
# =========================
def plot_hybrid_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    ax.text(2, 5, "Classical SVM", bbox=dict(fc="orange"))
    ax.text(8, 5, "Quantum SVM", bbox=dict(fc="lightgreen"))

    ax.text(2, 3, "Optimization + Prediction")
    ax.text(8, 3, "Quantum Kernel + Measurement")

    ax.arrow(3, 5, 3, 0, head_width=0.2)

    plt.title("Classical vs Quantum Hybrid Model")
    plt.savefig("outputs/hybrid_architecture.png", dpi=300)
    plt.close()

# ===== 4. ROC 曲线 (经典 SVM 专用) =====
def plot_roc(svm_model, X_test, y_test):
    fig = plt.figure(figsize=(6, 5))
    fig.canvas.manager.set_window_title("经典 SVM - ROC 性能曲线")
    
    RocCurveDisplay.from_estimator(svm_model, X_test, y_test, name="最优传统 SVM", ax=plt.gca())
    
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("经典 SVM ROC 性能曲线")
    plt.xlabel("假正例率 (False Positive Rate)")
    plt.ylabel("真正例率 (True Positive Rate)")
    plt.tight_layout()
    plt.show()

# =========================
# 一键运行
# =========================
if __name__ == "__main__":
    print("Generating theory plots...")

    plot_kernel_mapping()
    print("✓ kernel_mapping.png")

    plot_qsvm_flowchart()
    print("✓ qsvm_flowchart.png")

    plot_hybrid_architecture()
    print("✓ hybrid_architecture.png")

    print("DONE.")