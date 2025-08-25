from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib, os

def train_models(X_train, y_train):
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "svm": SVC(kernel="linear", probability=True, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def save_best_model(models, X_test, y_test, encoder, scaler):
    os.makedirs("models", exist_ok=True)
    results = {}
    for name, model in models.items():
        acc = evaluate_model(model, X_test, y_test)
        results[name] = acc
        print(f"{name} → accuracy = {acc:.2f}")

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    joblib.dump(best_model, f"models/{best_model_name}_model.pkl")
    joblib.dump(encoder, "models/encoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"\n✅ Best model saved: {best_model_name} with acc = {results[best_model_name]:.2f}")
    return best_model_name
