from flask import Flask, request, jsonify
import core

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Bienvenue sur l’API OrienGO 🚀"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # ex: {"skill": "Python", "experience": 2}
    if not data:
        return jsonify({"error": "Aucune donnée envoyée"}), 400
    try:
        job = core.predict_job(data)
        return jsonify({"predicted_job": job})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
