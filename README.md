# AI Test Case Classifier using Topic Modeling

A Streamlit app to classify industrial automation test cases using LDA Topic Modeling.  
It allows users to upload test cases in CSV format, runs NLP processing and LDA,  
and shows topic-labeled outputs with interactive graph visualization.

## Features
- Upload test case `.csv`
- Clean and preprocess text
- Apply LDA using scikit-learn
- Interactive topic graph using PyVis
- Streamlit-based UI

## How to Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
