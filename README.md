#  AI Test Case Classifier

An AI-powered tool that uses **Natural Language Processing (NLP)** and **LDA topic modeling** to classify and visualize **industrial test cases**.  
Built with Python, this project features an interactive **Streamlit web app** and **Pyvis-based graph visualization** to explore test case clusters and similarities.

---

##  Features

-  **Preprocessing & Similarity Scoring**  
  Cleans and vectorizes test cases, computes cosine similarity between them.

-  **Topic Modeling using LDA (scikit-learn)**  
  Identifies latent topic clusters from test case descriptions.

-  **Streamlit Web App**  
  Allows users to upload `.docx` files and explore topic-wise grouping.

-  **Interactive Graph Visualization (Pyvis)**  
  View test case relationships and clusters visually using dynamic graphs.

---

##  Tech Stack

- **Python 3.10+**
- **scikit-learn** – for LDA topic modeling  
- **NLTK / spaCy** – for text preprocessing  
- **Pyvis + NetworkX** – for graph-based visualization  
- **Streamlit** – for UI/UX and user interaction

---

##  How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ai-test-case-classifier.git
   cd ai-test-case-classifier


