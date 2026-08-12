import streamlit as st
import pickle

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(page_title="Spam Mail Detector", page_icon="📧", layout="wide")

# ------------------------------------------------------------
# Load model and vectorizer
# ------------------------------------------------------------
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.title("📧 About")
    st.markdown("**Spam Mail Detector**")
    st.write("This app uses a **Logistic Regression** model with TF-IDF text features to classify messages as:")
    st.markdown("- 🟢 **Ham** (Genuine message)")
    st.markdown("- 🔴 **Spam**")

    st.markdown("---")
    st.markdown("**Model Accuracy:** ~96.7%")
    st.markdown("**Technique:** TF-IDF + Logistic Regression")

    st.markdown("---")
    st.markdown("**How to Use**")
    st.markdown(
        "1. Paste an email or message\n"
        "2. Click Check Message\n"
        "3. View the prediction and confidence score"
    )

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("📧 Spam Mail Detector")
st.markdown("Paste an email or message below to check if it's spam or a genuine (ham) message")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["📝 Check Message", "ℹ️ Model Info"])

with tab1:
    message = st.text_area("Message text", height=200, placeholder="Paste your email/message here...")

    predict_clicked = st.button("🔍 Check Message", type="primary", use_container_width=True)

    if predict_clicked:
        if not message.strip():
            st.warning("⚠️ Please enter a message to check.")
        else:
            input_features = vectorizer.transform([message])
            prediction = model.predict(input_features)[0]
            proba = model.predict_proba(input_features)[0]

            if prediction == 1:
                st.success(f"✅ This looks like a genuine (Ham) message — confidence: {proba[1]*100:.2f}%")
            else:
                st.error(f"🚫 This looks like Spam — confidence: {proba[0]*100:.2f}%")

with tab2:
    st.subheader("Model Information")
    st.markdown("""
    - **Model Type:** Logistic Regression
    - **Text Features:** TF-IDF vectorization (English stop words removed)
    - **Dataset:** SMS Spam Collection dataset (labeled spam/ham messages)
    - **Test Accuracy:** ~96.7%
    """)
    st.info("This model generalizes based on word patterns learned during training — it may miss spam using unusual vocabulary not seen in the training data.")
# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ This application is an educational Machine Learning "
    "project. Its predictions should not be considered a "
    "guarantee that an email is safe or malicious."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    Spam Mail Detection System
    <br>
    Developed By Apurva Joshi 
    </div>
    """,
    unsafe_allow_html=True
)
