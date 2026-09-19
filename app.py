
import streamlit as st
import pickle
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords

# 1. Page Configuration
st.set_page_config(
    page_title="AI Sentiment Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NLTK Stopwords Setup
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words("english"))

# Custom CSS for Modern Dark UI Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-size: 16px !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.5) !important;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6) !important;
    }

    /* Tag Badges for Keywords */
    .pos-tag {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 4px 10px;
        border-radius: 8px;
        margin: 3px;
        display: inline-block;
        font-weight: 600;
    }
    .neg-tag {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 8px;
        margin: 3px;
        display: inline-block;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# 3. Accurate Explainability Function (Perturbation Impact Analysis)
def explain_prediction(text, vectorizer, model):
    words = text.split()
    if not words:
        return []

    base_vec = vectorizer.transform([text])
    
    if hasattr(model, "predict_proba"):
        base_pos_prob = model.predict_proba(base_vec)[0][1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(base_vec)[0]
        base_pos_prob = 1 / (1 + np.exp(-decision))
    else:
        base_pos_prob = 0.5

    word_impacts = []
    unique_words = list(set(words))

    for word in unique_words:
        text_without_word = " ".join([w for w in words if w != word])
        vec_without = vectorizer.transform([text_without_word])

        if hasattr(model, "predict_proba"):
            prob_without = model.predict_proba(vec_without)[0][1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(vec_without)[0]
            prob_without = 1 / (1 + np.exp(-decision))
        else:
            prob_without = 0.5

        # impact > 0 means word pushed sentiment towards Positive
        # impact < 0 means word pushed sentiment towards Negative
        impact = base_pos_prob - prob_without
        
        if abs(impact) > 0.0001:
            word_impacts.append((word, impact))

    return sorted(word_impacts, key=lambda x: abs(x[1]), reverse=True)

# 4. Resource Loader
@st.cache_resource
def load_resources():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "SVM": "svm.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "XGBoost": "xgboost.pkl"
    }
    
    for name, file_path in model_files.items():
        try:
            with open(file_path, "rb") as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            pass
            
    return vectorizer, models

try:
    tfidf, models = load_resources()
except Exception as e:
    st.error(f"⚠️ Model Initialization Error: {e}")
    st.stop()

# 5. Sidebar Control
st.sidebar.image("https://img.icons8.com/fluency/96/movie-projector.png", width=70)
st.sidebar.title("⚙️ Engine Control")
selected_model_name = st.sidebar.selectbox(
    "Active Classifier Model",
    list(models.keys()) if models else ["No Models Found"]
)

# 6. Main Hero Section
st.title("🎬 Movie Review Sentiment AI")
st.markdown("##### Real-time Sentiment Analysis & Explainable AI (XAI)")
st.write("")

col_input, col_stats = st.columns([2, 1], gap="medium")

with col_input:
    user_input = st.text_area(
        "✨ Enter Your Movie Review",
        placeholder="e.g., The plot was terrible and boring, but the acting was absolutely fantastic and brilliant!",
        height=180
    )
    analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)

with col_stats:
    st.markdown("### 📊 Engine Status")
    st.metric(label="Selected Model", value=selected_model_name)
    st.metric(label="Vectorizer Features", value="5,000 TF-IDF")

# 7. Prediction Logic & Visual Output
if analyze_btn:
    if user_input.strip() != "":
        if selected_model_name in models:
            cleaned_text = clean_text(user_input)
            vec_text = tfidf.transform([cleaned_text])
            
            model = models[selected_model_name]
            prediction = model.predict(vec_text)[0]
            
            # Confidence Probability
            pos_prob = 0.5
            neg_prob = 0.5
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vec_text)[0]
                neg_prob = float(probs[0])
                pos_prob = float(probs[1])
            elif hasattr(model, "decision_function"):
                decision = float(model.decision_function(vec_text)[0])
                pos_prob = 1 / (1 + np.exp(-decision))
                neg_prob = 1 - pos_prob

            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            
            if prediction == 1:
                m_col1.metric("Predicted Sentiment", "POSITIVE 🎉", delta="Positive Match")
                m_col2.metric("Positive Confidence", f"{pos_prob * 100:.1f}%")
                m_col3.metric("Negative Score", f"{neg_prob * 100:.1f}%")
                st.success("🌟 **Positive Review Detected!** The model identified favorable wording.")
                st.progress(pos_prob)
            else:
                m_col1.metric("Predicted Sentiment", "NEGATIVE ❌", delta="-Negative Match", delta_color="inverse")
                m_col2.metric("Negative Confidence", f"{neg_prob * 100:.1f}%")
                m_col3.metric("Positive Score", f"{pos_prob * 100:.1f}%")
                st.error("💔 **Negative Review Detected!** The model identified unfavorable wording.")
                st.progress(neg_prob)

            # 8. Explainable AI Section (Influential Words)
            st.markdown("---")
            st.subheader("🔍 Why did the AI make this decision?")
            st.write("Below are the key influential words found in your text that affected the decision:")

            explanation = explain_prediction(cleaned_text, tfidf, model)
            
            if explanation:
                pos_words = [word for word, impact in explanation if impact > 0]
                neg_words = [word for word, impact in explanation if impact < 0]
                
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    st.markdown("##### 🟢 Words driving POSITIVE sentiment:")
                    if pos_words:
                        html_pos = "".join([f'<span class="pos-tag">+{w}</span>' for w in pos_words[:8]])
                        st.markdown(html_pos, unsafe_allow_html=True)
                    else:
                        st.caption("No strong positive indicators found.")

                with exp_col2:
                    st.markdown("##### 🔴 Words driving NEGATIVE sentiment:")
                    if neg_words:
                        html_neg = "".join([f'<span class="neg-tag">-{w}</span>' for w in neg_words[:8]])
                        st.markdown(html_neg, unsafe_allow_html=True)
                    else:
                        st.caption("No strong negative indicators found.")
            else:
                st.info("No significant single-word impacts detected for this review.")
                
        else:
            st.error("⚠️ Selected model is not available.")
    else:
        st.warning("⚠️ Please enter a review text first.")