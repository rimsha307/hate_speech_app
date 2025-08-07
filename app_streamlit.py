import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

import re
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip()

    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)

    return cleaned_text


# ----- APP CONFIG -----
st.set_page_config(page_title="Hate Speech Classifier", layout="wide", page_icon="🧠")
st.title("🧠 Hate Speech Detection App")

# ----- LOAD MODEL -----
model = joblib.load("model.pkl")  # Replace with your model file
vectorizer = joblib.load("vectorizer.pkl")    # Replace with your vectorizer

# ----- SESSION STATE FOR PIE CHART COUNTS -----
if 'hate_count' not in st.session_state:
    st.session_state.hate_count = 0
    st.session_state.neutral_count = 0

# ----- INPUT + PIE CHART LAYOUT -----
left_col, right_col = st.columns([2, 1])

with left_col:
    user_input = st.text_area("Enter Text to Classify:", height=150)

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            cleaned_text = preprocess_text(user_input)
            text_vector = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_vector)[0]

            if prediction == 0:
                st.session_state.hate_count += 1
                st.error("⚠️ This text is classified as: **Hate/Offensive**")
            else:
                st.session_state.neutral_count += 1
                st.success("✅ This text is classified as: **Neutral**")

with right_col:
    if st.session_state.hate_count + st.session_state.neutral_count > 0:
        st.subheader("📊 Prediction Distribution")
        labels = ['Hate/Offensive', 'Neutral']
        sizes = [st.session_state.hate_count, st.session_state.neutral_count]
        colors = ['#ff4b4b', '#4CAF50']

        fig, ax = plt.subplots(figsize=(3, 3), facecolor='black')
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

# ----- SIDEBAR FOR CSV UPLOAD -----
st.sidebar.header("📄 Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload a CSV File", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
        st.sidebar.write("Preview:", df.head())

        col_options = df.columns.tolist()
        text_col = st.sidebar.selectbox("Select column with text", col_options)

        if st.sidebar.button("Classify Batch"):
            texts = df[text_col].astype(str)
            cleaned_texts = texts.apply(preprocess_text)
            vectors = vectorizer.transform(cleaned_texts)
            preds = model.predict(vectors)
            df['Prediction'] = preds
            df['Prediction Label'] = df['Prediction'].map({0: 'Hate/Offensive', 1: 'Neutral'})
            st.subheader("📋 Batch Predictions")
            st.dataframe(df[[text_col, 'Prediction Label']])

            # Download CSV
            output = BytesIO()
            df.to_csv(output, index=False)
            st.download_button("📥 Download Predictions as CSV", data=output.getvalue(),
                               file_name='predictions.csv', mime='text/csv')
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
