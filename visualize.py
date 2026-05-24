import matplotlib.pyplot as plt
import numpy as np
# 必须包含这一行，缺一不可！
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# 统一设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# ===== 1. 数据分布 =====
def plot_data(X, y):
    fig = plt.figure(figsize=(6, 5))
    fig.canvas.manager.set_window_title("1 - 测试集数据分布") 
    plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolors='k', s=40)
    plt.title("双月数据集真实分布 (测试集)")
    plt.xlabel("特征 1 (标准化)")
    plt.ylabel("特征 2 (标准化)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ===== 2. 完美对齐大纲：决策边界并排对比图 (1行2列) =====
def plot_side_by_side_boundaries(svm_model, qsvm_model, X_train, y_train, X_test, y_test, svm_acc, qsvm_acc, qsvm_circuit_func):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title("2 - 经典 RBF SVM vs QSVM 决策边界并排对比图")
    
    # 建立网格 (经典用高精度，量子用密集网格会极慢，这里做平衡取0.25)
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    
    # 1. 绘制经典 SVM 边界
    xx_c, yy_c = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z_c = svm_model.predict(np.c_[xx_c.ravel(), yy_c.ravel()]).reshape(xx_c.shape)
    ax1.contourf(xx_c, yy_c, Z_c, alpha=0.25, cmap="coolwarm")
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors='k', s=40)
    ax1.set_title(f"经典 SVM (RBF核) - 准确率: {svm_acc*100:.2f}%")
    ax1.set_xlabel("特征 1 (标准化)")
    ax1.set_ylabel("特征 2 (标准化)")

    # 2. 绘制量子 QSVM 边界
    print("\n[渲染器] 正在现场计算背景网格的量子核矩阵以绘制并排对比图...")
    xx_q, yy_q = np.meshgrid(np.arange(x_min, x_max, 0.3), np.arange(y_min, y_max, 0.3))
    grid_q = np.c_[xx_q.ravel(), yy_q.ravel()]
    K_grid = np.zeros((len(grid_q), len(X_train)))
    for i in range(len(grid_q)):
        for j in range(len(X_train)):
            K_grid[i, j] = qsvm_circuit_func(grid_q[i], X_train[j])[0]
    Z_q = qsvm_model.predict(K_grid).reshape(xx_q.shape)
    
    ax2.contourf(xx_q, yy_q, Z_q, alpha=0.25, cmap="coolwarm")
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors='k', s=40)
    ax2.set_title(f"量子 QSVM (角度编码) - 准确率: {qsvm_acc*100:.2f}%")
    ax2.set_xlabel("特征 1 (标准化)")
    ax2.set_ylabel("特征 2 (标准化)")
    
    plt.tight_layout()
    plt.show()

# ===== 3. 完美对齐大纲：经典 RBF 核 vs 量子核矩阵热力图对比 =====
def plot_kernel_matrices_comparison(K_classic, K_quantum, y_train):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.canvas.manager.set_window_title("3 - 经典 RBF 核 vs 量子核矩阵热力图对比")
    
    sort_idx = np.argsort(y_train)
    K_c_sorted = K_classic[sort_idx][:, sort_idx]
    K_q_sorted = K_quantum[sort_idx][:, sort_idx]
    
    # 统一归一化刻度到 0~1 方便直接对比
    im1 = ax1.imshow(K_c_sorted, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax1.set_title('经典 RBF 核矩阵 (Gram Matrix)')
    fig.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(K_q_sorted, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax2.set_title('量子核矩阵 (Angle Embedding)')
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

# ===== 4. 完美对齐大纲：性能指标对比柱状图 =====
def plot_performance_bars(svm_acc, svm_f1, qsvm_acc, qsvm_f1):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.canvas.manager.set_window_title("4 - 性能指标对比柱状图")
    
    labels = ['准确率 (Accuracy)', '宏平均 F1-Score']
    svm_scores = [svm_acc, svm_f1]
    qsvm_scores = [qsvm_acc, qsvm_f1]
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, svm_scores, width, label='经典 SVM (RBF)', color='#4682B4')
    rects2 = ax.bar(x + width/2, qsvm_scores, width, label='量子 QSVM', color='#3CB371')
    
    ax.set_ylabel('得分百分比')
    ax.set_title('模型量化性能核心指标对比')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # 柱头上自动标注具体数字
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    
    plt.tight_layout()
    plt.show()

# ===== 5. 混淆矩阵 =====
def plot_cm(y_true, y_pred, title):
    fig = plt.figure(figsize=(5, 5))
    fig.canvas.manager.set_window_title(f"混淆矩阵 - {title}")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=plt.gca())
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
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