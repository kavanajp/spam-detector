from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Detect URLs
def detect_url(text):
    return re.findall(r'(https?://\S+)', text)

# Home
@app.route("/")
def home():
    return render_template("index.html")

# Predict text
@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message")

    if not message:
        return render_template("index.html", prediction_text="⚠ Enter message")

    cleaned = clean_text(message)
    transformed = vectorizer.transform([cleaned])

    prediction = model.predict(transformed)[0]
    probs = model.predict_proba(transformed)[0]

    spam_prob = round(probs[1] * 100, 2)
    ham_prob = round(probs[0] * 100, 2)

    urls = detect_url(message)

    if prediction == 1:
        result = f"🚫 SPAM ({spam_prob}%)"
    else:
        result = f"✅ SAFE ({ham_prob}%)"

    if urls:
        result += " ⚠ Suspicious link detected"

    return render_template("index.html", prediction_text=result)

# File upload (FIXED)
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", prediction_text="⚠ No file selected")

        text = file.read().decode("utf-8")

        cleaned = clean_text(text)
        transformed = vectorizer.transform([cleaned])

        prediction = model.predict(transformed)[0]

        if prediction == 1:
            result = "🚫 SPAM (File Content)"
        else:
            result = "✅ SAFE (File Content)"

        return render_template("index.html", prediction_text=result)

    except:
        return render_template("index.html", prediction_text="⚠ File error")

# Run
if __name__ == "__main__":
    app.run(debug=True)