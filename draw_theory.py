import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from qiskit.circuit.library import ZZFeatureMap
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 1. 核方法高维映射示意图
def plot_kernel_mapping():
    np.random.seed(42)
    theta = np.linspace(0, np.pi, 50)
    x1 = np.cos(theta) + np.random.normal(0, 0.1, 50)
    y1 = np.sin(theta) + np.random.normal(0, 0.1, 50)
    x2 = np.cos(theta + np.pi) + np.random.normal(0, 0.1, 50)
    y2 = np.sin(theta + np.pi) + np.random.normal(0, 0.1, 50)
    
    z1 = x1**2 + y1**2
    z2 = x2**2 + y2**2
    
    fig = plt.figure(figsize=(16, 7))
    
    ax1 = fig.add_subplot(121)
    ax1.scatter(x1, y1, c='blue', label='类别0', edgecolors='k', s=50)
    ax1.scatter(x2, y2, c='red', label='类别1', edgecolors='k', s=50)
    ax1.set_title('低维特征空间（非线性可分）', fontsize=14, pad=15)
    ax1.set_xlabel('特征1', fontsize=12)
    ax1.set_ylabel('特征2', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x1, y1, z1, c='blue', label='类别0', edgecolors='k', s=50)
    ax2.scatter(x2, y2, z2, c='red', label='类别1', edgecolors='k', s=50)
    
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))
    zz = np.ones_like(xx) * 0.5
    ax2.plot_surface(xx, yy, zz, alpha=0.5, color='green')
    
    ax2.set_title('高维希尔伯特空间（线性可分）', fontsize=14, pad=15)
    ax2.set_xlabel('特征1', fontsize=12)
    ax2.set_ylabel('特征2', fontsize=12)
    ax2.set_zlabel('映射特征3', fontsize=12)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('kernel_mapping.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. QSVM算法流程图
def plot_qsvm_flowchart():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    box_style = dict(boxstyle="round,pad=0.5", fc="lightblue", ec="blue", lw=2)
    quantum_box = dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="green", lw=2)
    classic_box = dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", lw=2)
    
    steps = [
        (1, 5, "数据准备", box_style),
        (3, 5, "量子编码", quantum_box),
        (5, 5, "核矩阵计算", quantum_box),
        (7, 5, "经典优化", classic_box),
        (9, 5, "预测输出", box_style)
    ]
    
    for x, y, text, style in steps:
        ax.text(x, y, text, ha="center", va="center", fontsize=12, bbox=style)
    
    for i in range(len(steps)-1):
        x1, y1 = steps[i][0], steps[i][1]
        x2, y2 = steps[i+1][0], steps[i+1][1]
        ax.arrow(x1+0.8, y1, x2-x1-1.6, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    ax.text(4, 3, "量子处理器(QPU)执行", ha="center", va="center", fontsize=12, color='green')
    ax.text(7, 3, "经典处理器(CPU)执行", ha="center", va="center", fontsize=12, color='orange')
    
    ax.add_patch(patches.FancyBboxPatch((2.2, 3.8), 3.6, 1.8, boxstyle="square,pad=0.1", ec="green", fc="none", lw=2))
    ax.add_patch(patches.FancyBboxPatch((6.2, 3.8), 1.6, 1.8, boxstyle="square,pad=0.1", ec="orange", fc="none", lw=2))
    
    plt.title('量子支持向量机(QSVM)算法流程', fontsize=16, pad=20)
    plt.savefig('qsvm_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. 量子-经典混合计算协作示意图
def plot_hybrid_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    classic_box = patches.FancyBboxPatch((1, 1), 3, 6, boxstyle="round,pad=0.2", fc="lightyellow", ec="orange", lw=3)
    ax.add_patch(classic_box)
    ax.text(2.5, 7, "经典计算机(CPU/GPU)", ha="center", va="center", fontsize=14, fontweight='bold')
    
    classic_modules = [
        (2.5, 5.5, "数据预处理"),
        (2.5, 4, "SVM优化器"),
        (2.5, 2.5, "结果分析与可视化")
    ]
    
    for x, y, text in classic_modules:
        ax.text(x, y, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round", fc="white", ec="orange"))
    
    quantum_box = patches.FancyBboxPatch((6, 1), 3, 6, boxstyle="round,pad=0.2", fc="lightgreen", ec="green", lw=3)
    ax.add_patch(quantum_box)
    ax.text(7.5, 7, "量子处理器(QPU)", ha="center", va="center", fontsize=14, fontweight='bold')
    
    quantum_modules = [
        (7.5, 5.5, "量子态编码"),
        (7.5, 4, "量子电路执行"),
        (7.5, 2.5, "量子测量")
    ]
    
    for x, y, text in quantum_modules:
        ax.text(x, y, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round", fc="white", ec="green"))
    
    ax.arrow(4.2, 5.5, 1.6, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.text(5, 5.7, "经典数据→量子态", ha="center", va="bottom", fontsize=10)
    
    ax.arrow(5.8, 2.5, -1.6, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.text(5, 2.3, "测量结果→经典数据", ha="center", va="top", fontsize=10)
    
    ax.text(5, 0.5, "量子优势：高效计算高维量子核函数\n经典优势：擅长优化和后处理", 
            ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round", fc="lightgray"))
    
    plt.title('量子-经典混合计算架构', fontsize=16, pad=20)
    plt.savefig('hybrid_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. ZZFeatureMap量子电路图（含不同纠缠策略对比）
def plot_zz_feature_map_comparison():
    fm_linear = ZZFeatureMap(feature_dimension=2, reps=1, entanglement='linear')
    fm_full = ZZFeatureMap(feature_dimension=2, reps=1, entanglement='full')
    
    fig1 = fm_linear.draw(output="mpl", style="iqp", fold=-1)
    fig1.suptitle("ZZFeatureMap(线性纠缠, reps=1)", y=1.02, fontsize=14)
    fig1.savefig("zz_featuremap_linear.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    
    fig2 = fm_full.draw(output="mpl", style="iqp", fold=-1)
    fig2.suptitle("ZZFeatureMap(全纠缠, reps=1)", y=1.02, fontsize=14)
    fig2.savefig("zz_featuremap_full.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

# 5. 双月数据集样本分布图
def plot_moons_dataset():
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], label="类别0", c="blue", edgecolors="k", s=50)
    plt.scatter(X[y==1, 0], X[y==1, 1], label="类别1", c="red", edgecolors="k", s=50)
    
    plt.xlabel("特征1(标准化)", fontsize=12)
    plt.ylabel("特征2(标准化)", fontsize=12)
    plt.title("双月数据集分布(噪声=0.15)", fontsize=14, pad=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig("moons_dataset.png", dpi=300, bbox_inches="tight")
    plt.close()

# 运行所有第一阶段的图
if __name__ == "__main__":
    print("正在生成第一阶段理论图...")
    plot_kernel_mapping()
    print("✓ 核方法高维映射示意图已生成")
    plot_qsvm_flowchart()
    print("✓ QSVM算法流程图已生成")
    plot_hybrid_architecture()
    print("✓ 量子-经典混合架构图已生成")
    plot_zz_feature_map_comparison()
    print("✓ ZZFeatureMap电路图已生成")
    plot_moons_dataset()
    print("✓ 双月数据集分布图已生成")
    print("\n第一阶段完成！所有图片已保存到当前文件夹")
