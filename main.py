import pandas as pd
from src.preprocessing import clean_text
from src.lda_model import apply_lda_sklearn
from src.graph_builder import build_graph

if __name__ == "__main__":
    # Example local run with sample cases (for testing only)
    test_cases = [
        "Check if emergency stop is triggered when overheating",
        "Verify digital output signal when start button is pressed",
        "Test Ethernet connection between PLC and HMI",
        "Ensure PLC communication resumes after restart",
        "Check response when the safety lock is disengaged"
    ]

    df = pd.DataFrame({"test_case": test_cases})
    df["clean_text"] = df["test_case"].apply(clean_text)

    topics, topic_ids = apply_lda_sklearn(df["clean_text"])
    df["topic_id"] = topic_ids

    print(df)
    build_graph(df, topic_column="topic_id")
