import streamlit as st
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit config
st.set_page_config(page_title="Hate Speech Detector", layout="wide")

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title
st.title("🧠 Hate Speech Classifier")
st.write("Enter a sentence to check whether it's Hate/Offensive or Neutral.")

# Track prediction counts
if 'hate_count' not in st.session_state:
    st.session_state.hate_count = 0
if 'neutral_count' not in st.session_state:
    st.session_state.neutral_count = 0

# Input text box
user_input = st.text_area("Enter Text:", height=150 , width = 300)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize and predict
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]

        # Display result and update counts
        if prediction == 0:
            st.session_state.hate_count += 1
            st.error("⚠️ This text is classified as: **Hate/Offensive**")
        else:
            st.session_state.neutral_count += 1
            st.success("✅ This text is classified as: **Neutral**")

        # Visualization columns
        st.markdown("---")
        col1, col2 = st.columns(2)

        # ----- PIE CHART -----
        with col1:
            st.subheader("📊 Prediction Distribution")
            labels = ['Hate/Offensive', 'Neutral']
            sizes = [st.session_state.hate_count, st.session_state.neutral_count]
            colors = ['#ff4b4b', '#4CAF50']

            fig, ax = plt.subplots(figsize=(1.5, 1.5), facecolor='black')
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        # ----- WORD CLOUD -----
        with col2:
            st.subheader("🌥 Sample Word Cloud")
            sample_text = """
            hate speech offensive violence abuse slur racism sexism hate anger insult harassment 
            neutral peaceful love unity respect tolerance helpful safe informative
            """
            wordcloud = WordCloud(width=400, height=300, background_color= 'black', colormap='Set2').generate(sample_text)

            fig_wc, ax_wc = plt.subplots(figsize=(3, 3))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            fig_wc.patch.set_facecolor('black')  # remove border
            st.pyplot(fig_wc)
