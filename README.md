# fake-news-detection-rag-llm
Fake news detection using RAG and LLMs with evidence-based classification and explainable AI

This project detects whether a news article is fake or real using Retrieval-Augmented Generation (RAG) and Large Language Models.

## 🚀 Features
- Retrieves relevant articles using vector similarity (FAISS)
- Uses LLM for classification and explanation
- Displays supporting evidence for transparency

## 📊 Results
- Accuracy improved from 65% → 88%
- Precision: 0.94 | Recall: 0.81 | F1-score: 0.87

## 🛠 Tech Stack
Python, FAISS, Sentence Transformers, OpenAI, Streamlit

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
