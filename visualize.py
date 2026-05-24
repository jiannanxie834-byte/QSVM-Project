import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# ==========================================
# 全局设置：统一中文字体，防止图片中的汉字变方块
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows默认黑体，如果是Mac可以换成 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. 核方法高维映射示意图 (含 3D 分离超平面)
# ==========================================
def plot_kernel_mapping():
    np.random.seed(42)
    # 模拟 150 样本数据分布形态
    theta = np.linspace(0, 2*np.pi, 100)
    r1 = 0.5 + np.random.normal(0, 0.08, 100)
    r2 = 1.0 + np.random.normal(0, 0.08, 100)
    
    # 类别 0 (内圈)
    x1, y1 = r1 * np.cos(theta), r1 * np.sin(theta)
    z1 = x1**2 + y1**2  
    
    # 类别 1 (外圈)
    x2, y2 = r2 * np.cos(theta), r2 * np.sin(theta)
    z2 = x2**2 + y2**2

    fig = plt.figure(figsize=(13, 6))

    # 左子图：低维非线性
    ax1 = fig.add_subplot(121)
    ax1.scatter(x1, y1, c='#4682B4', label='类别 0 (内圈)')
    ax1.scatter(x2, y2, c='#CD5C5C', label='类别 1 (外圈)')
    ax1.set_title("原始低维特征空间 (非线性不可分)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("特征 1")
    ax1.set_ylabel("特征 2")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # 右子图：高维希尔伯特空间
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x1, y1, z1, c='#4682B4', alpha=0.7)
    ax2.scatter(x2, y2, z2, c='#CD5C5C', alpha=0.7)
    
    # 绘制切分两类的分离超平面
    xx, yy = np.meshgrid(np.linspace(-1.3, 1.3, 10), np.linspace(-1.3, 1.3, 10))
    zz = np.ones_like(xx) * 0.55  
    ax2.plot_surface(xx, yy, zz, alpha=0.3, color='#3CB371')
    
    ax2.text(0, 0, 0.65, "最优分离超平面\n(Decision Plane)", color="green", fontsize=10, fontweight='bold')
    ax2.set_title("映射后的高维希尔伯特空间 (线性可分)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("特征 1")
    ax2.set_ylabel("特征 2")
    ax2.set_zlabel("映射特征 3: $\Phi(X)$")

    plt.tight_layout()
    plt.savefig("outputs/kernel_mapping.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 2. QSVM 算法流程图 (修复了箭头穿帮问题)
# ==========================================
def plot_qsvm_flowchart():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    steps = ["经典数据\n(Data)", "量子编码\n(Encoding)", "核矩阵计算\n(Kernel)", "经典优化\n(Optimization)", "预测输出\n(Prediction)"]
    x_coords = [1.2, 3.2, 5.2, 7.2, 9.2]
    
    colors = ["#F0F8FF", "#E6F2EA", "#E6F2EA", "#FFF8DC", "#F0F8FF"]
    edge_colors = ["#4682B4", "#2E8B57", "#2E8B57", "#FF8C00", "#4682B4"]

    for i, (text, x, c, ec) in enumerate(zip(steps, x_coords, colors, edge_colors)):
        # 步骤方框
        ax.text(x, 3, text, ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.6", fc=c, ec=ec, lw=1.5))
        
        # 精美连线箭头
        if i < len(steps) - 1:
            ax.annotate("", xy=(x_coords[i+1] - 0.7, 3), xytext=(x + 0.7, 3),
                        arrowprops=dict(arrowstyle="-|>", lw=2, color="gray", mutation_scale=15))

    plt.title("量子支持向量机 (QSVM) 算法核心工作流", fontsize=14, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/qsvm_flowchart.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. 经典 vs 量子 架构对比图 (彻底修复排版)
# ==========================================
def plot_hybrid_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    # 经典计算机
    classic_box = patches.FancyBboxPatch((0.8, 1.2), 3.2, 4.5, boxstyle="round,pad=0.2", fc="#FFF8DC", ec="#FF8C00", lw=2)
    ax.add_patch(classic_box)
    ax.text(2.4, 5.4, "经典计算机 (CPU / GPU)", ha="center", va="center", fontsize=12, fontweight='bold', color="#D2691E")
    
    ax.text(2.4, 4.2, "数据预处理与标准化\n(StandardScaler)", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#FF8C00"))
    ax.text(2.4, 2.8, "经典 SVM 优化求解器\n(网格搜索参数寻优)", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#FF8C00"))
    ax.text(2.4, 1.6, "最终分类预测与可视化", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#FF8C00"))

    # 量子处理器
    quantum_box = patches.FancyBboxPatch((6.0, 1.2), 3.2, 4.5, boxstyle="round,pad=0.2", fc="#E6F2EA", ec="#2E8B57", lw=2)
    ax.add_patch(quantum_box)
    ax.text(7.6, 5.4, "量子处理器 (QPU 模拟器)", ha="center", va="center", fontsize=12, fontweight='bold', color="#1E5D3A")
    
    ax.text(7.6, 4.2, "量子态角度编码\n(Angle Embedding)", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2E8B57"))
    ax.text(7.6, 2.8, "量子伴随线路执行\n(计算希尔伯特空间内积)", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2E8B57"))
    ax.text(7.6, 1.6, "量子态保真度测量\n(输出预计算量子核矩阵)", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2E8B57"))

    # 数据流双向箭头
    ax.annotate("", xy=(5.9, 4.2), xytext=(4.1, 4.2), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(5.0, 4.4, "输入经典特征", ha="center", va="bottom", fontsize=10, fontweight='bold')
    
    ax.annotate("", xy=(4.1, 2.8), xytext=(5.9, 2.8), arrowprops=dict(arrowstyle="->", lw=2, color="blue"))
    ax.text(5.0, 2.6, "传回量子核矩阵\n(Kernel Matrix)", ha="center", va="top", fontsize=10, color="blue", fontweight='bold')

    # 底层说明
    ax.text(5.0, 0.4, "经典优势：擅长高维凸优化与后处理\n量子优势：通过量子态空间实现无视维度的复杂非线性映射", 
            ha="center", va="center", fontsize=11, style='italic', bbox=dict(boxstyle="square,pad=0.5", fc="#F5F5F5", ec="gray", lw=1))

    plt.title("经典 - 量子混合计算协作架构图 (Hybrid Architecture)", fontsize=14, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/hybrid_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 主程序一键执行
# ==========================================
if __name__ == "__main__":
    # 自动检测并创建输出文件夹，防止报错
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    print("====== 开始生成高质量理论配图 ======")

    plot_kernel_mapping()
    print("✓ kernel_mapping.png (高维映射图 - 已修复 3D 超平面)")

    plot_qsvm_flowchart()
    print("✓ qsvm_flowchart.png (算法流程图 - 已修复排版与箭头)")

    plot_hybrid_architecture()
    print("✓ hybrid_architecture.png (混合架构图 - 已修复坐标溢出与数据流向)")

    print("====== 理论图表生成完毕！请在 outputs 文件夹中查看 ======")