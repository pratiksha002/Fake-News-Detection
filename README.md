# 📰 Fake News Detector (Machine Learning + NLP + Streamlit)

An end-to-end **Fake News Detection** web application that classifies a news article as **FAKE**, **REAL**, or **UNCERTAIN** using **Machine Learning + Natural Language Processing (NLP)**.  
The app is deployed using **Streamlit Cloud**.

---
## ⚠️ Disclaimer

This project is based on patterns learned from the training dataset.
It does not guarantee real-world truth and should be used for educational purposes only.
---
## 🚀 Live Demo
🔗 **Streamlit App:** https://fake-news-detection1902.streamlit.app

---

## 📌 GitHub Repository
🔗 **Repo Link:** https://github.com/pratiksha002/Fake-News-Detection.git
---

## 🎯 Project Objective
Fake news spreads quickly on the internet and can mislead people.  
This project aims to build a Machine Learning model that learns patterns from labeled news articles and predicts whether given news content is **real or fake**.

---

## 🧠 Workflow / Pipeline
1. **Dataset Collection** (Fake.csv + True.csv)
2. **Data Cleaning & Preprocessing**
   - Convert text to lowercase
   - Remove URLs
   - Remove numbers
   - Remove punctuation
   - Remove extra spaces
   - Remove stopwords
3. **Feature Extraction**
   - TF-IDF Vectorization
4. **Model Training**
   - Train multiple ML models
   - Compare performance
   - Select best model
5. **Model Evaluation**
   - Accuracy
   - Precision / Recall / F1-score
   - Confusion Matrix
6. **Model Saving**
   - Save trained model and vectorizer using `joblib`
7. **Deployment**
   - Streamlit Cloud Deployment

---

## 📂 Dataset
This project uses the **Fake and Real News Dataset** which contains labeled news articles.

Files used:
- `Fake.csv`
- `True.csv`

Each contains columns like:
- `title`
- `text`
- `subject`
- `date`

---

## 🔍 Models Used
Multiple algorithms were trained and tested to compare their performance:

✅ Logistic Regression  
✅ Linear SVM  
✅ Decision Tree  
✅ Random Forest  
✅ Ridge Classifier  

---

## 📊 Model Comparison (Accuracy)
Logistic Regression-0.994342
Linear SVM-0.998642
Decision Tree-0.996379
Random Forest-0.997171
Ridge Classifier-0.998642

---

## 🏆 Final Model Selected
✅ **Logistic Regression** was selected for deployment because:
- It provides **probability scores** using `predict_proba()`
- Works extremely well with TF-IDF features
- Supports **confidence-based prediction** (FAKE / REAL / UNCERTAIN)

Saved artifacts:
- `models/best_model.pkl`
- `models/tfidf_vectorizer.pkl`

---

## 🖥️ Web App Features
✅ Paste full news article text  
✅ Predict **FAKE / REAL / UNCERTAIN**  
✅ Displays:
- Confidence %
- Fake Probability
- Real Probability  
✅ Minimum word requirement for better prediction

---

## 🛠️ Tech Stack
**Languages**
- Python

**Libraries**
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

**Tools**
- VS Code
- Git & GitHub
- Streamlit Cloud

---

