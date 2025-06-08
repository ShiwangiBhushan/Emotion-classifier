# 💬 Emotion Detection Web App

This project is a web application built using Streamlit that detects emotions from text input using a machine learning model.

---

## 🧠 Approach Summary

1. **Preprocessing**:
   - Removed stopwords, user handles, and special characters using `neattext`.

2. **Modeling**:
   - Used a `Pipeline` with `CountVectorizer` and `LogisticRegression` from scikit-learn.
   - Trained the model on labeled text data to classify into predefined emotions.

3. **Deployment**:
   - Saved the trained model using `joblib`.
   - Built a Streamlit frontend to take user input and display the predicted emotion with probabilities.

---

## 📦 Dependencies

Make sure you have the following packages installed :

```txt
streamlit
pandas
numpy
altair
joblib
scikit-learn==1.6.1
neattext

