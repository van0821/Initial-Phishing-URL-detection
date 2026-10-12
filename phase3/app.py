import joblib
from flask import Flask, request


app = Flask(__name__)

bundle = joblib.load("phishing_rf.pkl")
model = bundle["model"]
feature_names = bundle["feature_names"]

@app.route("/")
def home():
    return {"status": "ok"}

@app.route("/predict",  methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]

    if len(features) != 30:
        return {"error": "Feature length must be 30"}, 400

    prediction = model.predict([features])[0]
    label = "钓鱼" if prediction == -1 else "正常"

    return {
        "prediction": int(prediction),
        "label": label,
    }



if __name__ == "__main__":
    app.run(debug=True, port=5000)