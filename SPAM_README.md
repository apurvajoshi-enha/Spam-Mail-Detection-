# Spam Mail Detector 📧

A machine learning web app that classifies email/text messages as **Spam** or **Ham (genuine)**. Built with a Logistic Regression model trained on labeled message data using TF-IDF text features, and deployed as an interactive Streamlit app.

**Live demo:** *(add your Streamlit Cloud link here once deployed)*

## Overview

Paste any email or message text into the app, and it predicts whether the message is spam, along with a confidence score.

## Tech Stack

- Python
- scikit-learn (Logistic Regression, TF-IDF Vectorizer)
- Streamlit

## Model Performance

Test Accuracy: ~96.7%

## Project Structure

```
├── app.py               # Streamlit web app
├── spam_model.pkl       # Trained Logistic Regression model
├── vectorizer.pkl       # Fitted TF-IDF vectorizer
└── requirements.txt     # Python dependencies
```

## Running Locally

1. Clone the repository
   ```
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Run the app
   ```
   streamlit run app.py
   ```

## Disclaimer

This tool is built for educational purposes and may not catch every spam pattern in real-world email.
