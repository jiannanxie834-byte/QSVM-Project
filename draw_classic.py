from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time

def train_classic_svm(X_train, X_test, y_train, y_test):
    """训练线性SVM和RBF核SVM"""
    results = {}
    
    # 训练线性SVM
    start_time = time.time()
    svm_linear = SVC(kernel='linear', random_state=42)
    svm_linear.fit(X_train, y_train)
    linear_train_time = time.time() - start_time
    
    y_pred_linear = svm_linear.predict(X_test)
    results['linear_accuracy'] = accuracy_score(y_test, y_pred_linear)
    results['linear_f1'] = f1_score(y_test, y_pred_linear)
    results['linear_train_time'] = linear_train_time
    results['linear_model'] = svm_linear
    
    # 训练RBF核SVM
    start_time = time.time()
    svm_rbf = SVC(kernel='rbf', random_state=42)
    svm_rbf.fit(X_train, y_train)
    rbf_train_time = time.time() - start_time
    
    y_pred_rbf = svm_rbf.predict(X_test)
    results['rbf_accuracy'] = accuracy_score(y_test, y_pred_rbf)
    results['rbf_f1'] = f1_score(y_test, y_pred_rbf)
    results['rbf_train_time'] = rbf_train_time
    results['rbf_model'] = svm_rbf
    
    return results
