import pandas as pd
from src.preprocessing import clean_text
from src.lda_model import apply_lda_sklearn
from src.graph_builder import build_graph

print("🔧 Setting up AI Test Case Classifier...")

# Load test cases
df = pd.read_csv("data/test_cases.csv")

# Clean text
df["clean_text"] = df["test_case"].apply(clean_text)
df.dropna(subset=["clean_text"], inplace=True)

# Apply LDA
print("🧠 Applying LDA Topic Modeling...")
topics, topic_ids = apply_lda_sklearn(df["clean_text"])
df["topic_id"] = topic_ids

# Show topics
print("\n📚 Topics Discovered:")
for idx, words in topics:
    print(f"Topic {idx}: {', '.join(words)}")

# Show sample result
print("\n📝 Test Cases with Assigned Topics:")
print(df[["test_case", "topic_id"]].head())

# Build interactive graph
print("\n🌐 Building interactive topic graph...")
build_graph(df, topic_column="topic_id")
print("\n✅ Graph generated: test_case_graph.html")



