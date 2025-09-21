from flask import Flask, render_template, request
import os
from model import train_model, predict_price

app = Flask(__name__)


if not os.path.exists("model.pkl"):
    train_model()

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None

    if request.method == "POST":
        try:
            data = request.form
            features = [
                float(data["income"]),
                float(data["house_age"]),
                float(data["rooms"]),
                float(data["bedrooms"]),
                float(data["population"])
            ]

            predicted_price = predict_price(features)

        except Exception as e:
            predicted_price = f"Error: {str(e)}"

    return render_template("index.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
