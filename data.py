from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_iris_data():
    """
    加载并预处理Iris二分类数据集
    返回：训练集特征、测试集特征、训练集标签、测试集标签
    """
    # 加载完整Iris数据集
    iris = datasets.load_iris()
    
    # 筛选前两类（标签0和1）和前两个特征，简化问题
    X = iris.data[:100, :2]  # 形状：(100, 2)
    y = iris.target[:100]    # 形状：(100,)，只有0和1两个标签
    
    # 按7:3比例划分训练集和测试集，保持类别分布一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 数据标准化（量子算法对数据范围非常敏感，必须做）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# 测试代码：运行这个文件可以验证数据加载是否正确
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_iris_data()
    print("训练集形状:", X_train.shape)
    print("测试集形状:", X_test.shape)
    print("训练集标签分布:", {0: sum(y_train==0), 1: sum(y_train==1)})
    print("测试集标签分布:", {0: sum(y_test==0), 1: sum(y_test==1)})
