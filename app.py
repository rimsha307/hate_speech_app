from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize app
app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"\brt\b", '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        input_text = request.form["text"]
        cleaned = clean_text(input_text)
        features = vectorizer.transform([cleaned])
        result = model.predict(features)[0]
        prediction = "Hate/Offensive" if result == 0 else "Neutral"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
