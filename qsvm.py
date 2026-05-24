import pennylane as qml
import numpy as np
from sklearn.svm import SVC

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# ===== 量子特征映射 =====
@qml.qnode(dev)
def quantum_kernel_circuit(x1, x2):
    qml.templates.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

# ===== kernel矩阵 =====
def quantum_kernel(X1, X2):
    kernel = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            kernel[i, j] = quantum_kernel_circuit(X1[i], X2[j])[0]
    return kernel

# ===== QSVM训练 =====
def run_qsvm(X_train, X_test, y_train):
    K_train = quantum_kernel(X_train, X_train)
    K_test = quantum_kernel(X_test, X_train)

    model = SVC(kernel="precomputed")
    model.fit(K_train, y_train)

    return model, K_train, K_test