from sklearn.svm import SVC

def run_svm(X_train, X_test, y_train):
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    model.fit(X_train, y_train)
    return model