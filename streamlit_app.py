import streamlit as st
import pandas as pd
from src.preprocessing import clean_text
from src.lda_model import apply_lda_sklearn
from src.graph_builder import build_graph
import os

st.set_page_config(page_title="AI Test Case Classifier", layout="wide")
st.title("🧠 AI Test Case Classifier with Topic Modeling")

uploaded_file = st.file_uploader("Upload Test Cases CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "test_case" not in df.columns:
        st.error("CSV must have a 'test_case' column.")
    else:
        st.subheader("📄 Sample Test Cases")
        st.dataframe(df.head())

        with st.spinner("Cleaning text and applying LDA..."):
            df["clean_text"] = df["test_case"].apply(clean_text)
            df.dropna(subset=["clean_text"], inplace=True)
            topics, topic_ids = apply_lda_sklearn(df["clean_text"], num_topics=5)
            df["topic_id"] = topic_ids

        st.success("LDA applied. Topics assigned.")

        st.subheader("📋 Test Cases with Assigned Topics")
        st.dataframe(df[["test_case", "topic_id"]].head(10))

        st.subheader("📚 Discovered Topics")
        for idx, words in topics:
            st.markdown(f"**Topic {idx}**: {', '.join(words)}")

        st.subheader("🌐 Visualize Topics as Graph")
        html_file = "test_case_graph.html"
        build_graph(df, topic_column="topic_id")

        if os.path.exists(html_file):
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)

        st.download_button(
            label="📥 Download Labeled CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="labeled_test_cases.csv",
            mime="text/csv"
        )
else:
    st.info("👆 Upload a .csv file to get started. Make sure it has a 'test_case' column.")
