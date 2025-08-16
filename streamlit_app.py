import streamlit as st
import pandas as pd
from src.preprocessing import clean_text
from src.lda_model import apply_lda_sklearn
from src.graph_builder import build_graph

st.set_page_config(page_title="AI Test Case Classifier", layout="wide")
st.title("🧪 AI Test Case Classifier using Topic Modeling")

# Upload file
uploaded_file = st.file_uploader("📂 Upload a .txt file with test cases (one per line):", type=["txt"])

if uploaded_file:
    # Read file content
    test_cases = uploaded_file.read().decode("utf-8").splitlines()
    df = pd.DataFrame({"test_case": test_cases})

    # Preprocessing
    df["clean_text"] = df["test_case"].apply(clean_text)

    # Topic Modeling
    st.subheader("🔍 Running LDA Topic Modeling...")
    topics, topic_ids = apply_lda_sklearn(df["clean_text"])
    df["topic_id"] = topic_ids

    # Show table
    st.subheader("📌 Classified Test Cases")
    st.dataframe(df, use_container_width=True)

    # Show graph
    st.subheader("🌐 Interactive Topic Graph")
    build_graph(df, topic_column="topic_id")

    # Download results
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results", csv, "classified_testcases.csv", "text/csv")

else:
    st.info("👆 Upload a `.txt` file to begin classification.")


