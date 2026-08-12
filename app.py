import streamlit as st
import pickle

# ------------------------------------------------------------
# Load model and vectorizer
# ------------------------------------------------------------
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Spam Mail Detector", page_icon="📧")
st.title("📧 Spam Mail Detector")
st.write("Paste an email or message below to check if it's spam or a genuine (ham) message.")

message = st.text_area("Message text", height=200, placeholder="Paste your email/message here...")

if st.button("Check Message"):
    if not message.strip():
        st.warning("Please enter a message to check.")
    else:
        input_features = vectorizer.transform([message])
        prediction = model.predict(input_features)[0]
        proba = model.predict_proba(input_features)[0]

        if prediction == 1:
            st.success(f"✅ This looks like a genuine (Ham) message — confidence: {proba[1]*100:.2f}%")
        else:
            st.error(f"🚫 This looks like Spam — confidence: {proba[0]*100:.2f}%")
