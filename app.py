import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# تحميل الموديلات
model_ct = joblib.load("cancer_type_model.pkl")
model_rl = joblib.load("risk_level_model_tuned.pkl")

le_ct = joblib.load("label_encoder_ct.pkl")
le_rl = joblib.load("label_encoder_rl.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    df = pd.DataFrame([data])

    # prediction cancer type
    pred_ct = model_ct.predict(df)[0]
    pred_ct = le_ct.inverse_transform([pred_ct])[0]

    # prediction risk level
    pred_rl = model_rl.predict(df)[0]
    pred_rl = le_rl.inverse_transform([pred_rl])[0]

    return {
        "Cancer_Type": pred_ct,
        "Risk_Level": pred_rl
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
