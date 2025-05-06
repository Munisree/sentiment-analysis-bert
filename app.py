import streamlit as st
from transformers import pipeline
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Set page config first
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

# Load model with caching
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

model = load_model()

# Custom HTML styling
st.markdown("""
    <h1 style='text-align: center;'>🎯 Sentiment Analysis with BERT</h1>
    <p style='text-align: center; color: gray;'>Analyze your reviews with a powerful transformer-based model.</p>
    <hr style='border: 1px solid #555;' />
""", unsafe_allow_html=True)

# Single Input Area
st.subheader("📝 Analyze Single Review")
user_input = st.text_area("Enter your review below:", height=140)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing with BERT..."):
            result = model(user_input)[0]
            sentiment = result['label']
            score = result['score']

            if 0.45 <= score <= 0.55:
                sentiment = "MIXED"
                emoji = "😐"
            elif sentiment == "POSITIVE":
                emoji = "😊"
            else:
                emoji = "😠"

            col1, col2 = st.columns(2)
            col1.success(f"**Sentiment:** {sentiment} {emoji}")
            col2.info(f"**Confidence:** `{score:.2f}`")

st.markdown("---")

# Batch Upload Section
st.subheader("📁 Upload a File for Batch Sentiment Analysis")
uploaded_file = st.file_uploader("Upload a Text File (.txt):", type=["txt"])

def display_sentiment_chart(result_df):
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = result_df['predicted_sentiment'].value_counts()

    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("❌ CSV must have a 'text' column.")
        else:
            st.success("✅ File loaded successfully.")
            st.dataframe(df.head())

            if st.button("🧠 Analyze Uploaded Reviews"):
                with st.spinner("Processing..."):
                    results = []
                    for review in df["text"]:
                        output = model(review)[0]
                        score = output['score']
                        label = output['label']
                        if 0.45 <= score <= 0.55:
                            label = "MIXED"
                        results.append({
                            "text": review,
                            "predicted_sentiment": label,
                            "confidence": round(score, 2)
                        })

                    result_df = pd.DataFrame(results)
                    st.success("✅ Analysis complete!")
                    st.dataframe(result_df)

                    # Chart
                    display_sentiment_chart(result_df)

                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results", data=csv, file_name="sentiment_results.csv", mime="text/csv")

    elif uploaded_file.name.endswith(".txt"):
        lines = uploaded_file.read().decode("utf-8").splitlines()
        st.success("✅ File loaded successfully.")
        st.write(lines[:5])

        if st.button("🧠 Analyze Uploaded Text File"):
            with st.spinner("Processing..."):
                results = []
                for review in lines:
                    output = model(review)[0]
                    score = output['score']
                    label = output['label']
                    if 0.45 <= score <= 0.55:
                        label = "MIXED"
                    results.append({
                        "text": review,
                        "predicted_sentiment": label,
                        "confidence": round(score, 2)
                    })

                result_df = pd.DataFrame(results)
                st.success("✅ Analysis complete!")
                st.dataframe(result_df)

                # Chart
                display_sentiment_chart(result_df)

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", data=csv, file_name="sentiment_results.csv", mime="text/csv")

# Footer
st.markdown("""
    <hr style='border: 0.5px solid #999;' />
    <p style='text-align: center; font-size: 0.9em; color: #888;'>Made with ❤️ using BERT and Streamlit · <a href='https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english' target='_blank'>Model Info</a></p>
""", unsafe_allow_html=True)
