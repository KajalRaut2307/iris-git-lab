import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def train(test_size=0.2, max_iter=1000):
    """
    function train:
    returns: model and accuracy
    """
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LogisticRegression(max_iter=max_iter, C=1.0)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    f1score = f1_score(y_te, model.predict(X_te), average="weighted")
    return model, acc, f1score

if __name__ == "__main__":
    model, acc, f1score = train()
    print(f"accuracy: {acc:.3f}")
    print(f"f1 score: {f1score:.3f}")
    joblib.dump(model, "model.joblib")