project: Jobs Recommendation System
description: 
  Système de recommandation de métiers basé sur le modèle RIASEC.
  Prédit le métier le plus adapté à un étudiant à partir de ses scores RIASEC,
  et lie le métier à une formation correspondante.

structure:
  data:
    - jobs.csv: Dataset métiers (category, description, education, job_market, RIASEC scores, salary_range)
    - trainings.csv: Dataset formations (name, type, duration, description)
  models:
    - best_model.pkl: meilleur modèle sauvegardé
    - encoder.pkl: encodeur des catégories
    - scaler.pkl: normaliseur des données
  notebooks:
    - JobRecommendation.ipynb: entraînement et tests
  utils:
    - preprocessing.py: prétraitement des données
  core.py: entraînement et sauvegarde des modèles
  app.py: (optionnel) API Flask pour prédictions
  requirements.txt: dépendances Python

models_used:
  - Logistic Regression (baseline)
  - Random Forest
  - KNN (K-Nearest Neighbors)
  - SVM (Support Vector Machine)

usage:
  training: |
    from utils.preprocessing import load_and_preprocess_data
    from core import train_models, save_best_model
    X_train, X_test, y_train, y_test, encoder, scaler = load_and_preprocess_data("data/jobs.csv")
    models = train_models(X_train, y_train)
    best_model_name, best_model = save_best_model(models, X_test, y_test, encoder, scaler)

  prediction: |
    import joblib, numpy as np
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")
    student_profile = np.array([[10, 20, 30, 15, 100, 25]])
    student_profile_scaled = scaler.transform(student_profile)
    predicted_class = model.predict(student_profile_scaled)[0]
    predicted_job = encoder.inverse_transform([predicted_class])[0]
    print("🎯 Métier recommandé:", predicted_job)

author: Douaa BOUSTANE

