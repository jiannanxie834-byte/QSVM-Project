from data import load_data
from svm import run_svm
from qsvm import run_qsvm, quantum_kernel_circuit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

from visualize import (
    plot_data,
    plot_side_by_side_boundaries,
    plot_kernel_matrices_comparison,
    plot_performance_bars,
    plot_cm,
    plot_roc
)

if __name__ == "__main__":
    print("====== 开始运行 经典与量子 SVM 混合实验流水线 ======")
    
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_data()

    # 2. 训练传统 SVM
    svm_model = run_svm(X_train, X_test, y_train)
    y_svm = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, y_svm)
    svm_f1 = f1_score(y_test, y_svm, average='macro')

    # 3. 训练量子 SVM
    qsvm_model, K_train, K_test = run_qsvm(X_train, X_test, y_train)
    y_qsvm = qsvm_model.predict(K_test)
    qsvm_acc = accuracy_score(y_test, y_qsvm)
    qsvm_f1 = f1_score(y_test, y_qsvm, average='macro')

    # 4. 打印文本报告
    print("\n【经典 SVM 详细报告】\n", classification_report(y_test, y_svm))
    print("【量子 QSVM 详细报告】\n", classification_report(y_test, y_qsvm))

    # 5. 计算经典 RBF 的训练集核矩阵用于对比
    gamma_val = svm_model.gamma if isinstance(svm_model.gamma, float) else 1.0 / X_train.shape[1]
    K_classic_train = rbf_kernel(X_train, X_train, gamma=gamma_val)

    # ===================================================
    # 顺序触发图表展示
    # ===================================================
    print("\n>>> 正在按规范生成图表窗口...")

    # 图 1：数据分布图
    plot_data(X_test, y_test)

    # 图 2：核心灵魂！并排决策边界对比图
    plot_side_by_side_boundaries(svm_model, qsvm_model, X_train, y_train, X_test, y_test, 
                                 svm_acc, qsvm_acc, quantum_kernel_circuit)

    # 图 3：底层原理！核矩阵热力图对比
    plot_kernel_matrices_comparison(K_classic_train, K_train, y_train)

    # 图 4：量化指标对比柱状图
    plot_performance_bars(svm_acc, svm_f1, qsvm_acc, qsvm_f1)

    # 附图：混淆矩阵
    plot_cm(y_test, y_svm, "SVM 混淆矩阵")
    plot_cm(y_test, y_qsvm, "QSVM 混淆矩阵")
    
    plot_roc(svm_model, X_test, y_test)
    print("\n====== 实验全工作流运行完毕，所有规划图表已齐活 ======")