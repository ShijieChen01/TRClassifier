import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
 
# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
CLASSIFIER_BUNDLE_PATH = "classification_bundle.pkl"
CORPUS_PATH = "WOS_data.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
#  Load Raw Data
# ─────────────────────────────────────────────────────────────────────────────
def load_corpus_data():
    """
    Load abstracts and labels from Excel.
    """
    data = pd.read_excel(CORPUS_PATH, converters={"Abstract": str})
    abstracts = data["Abstract"].fillna("").tolist()
    labels = data["Journal"].astype(str).reset_index(drop=True)
    return abstracts, labels

# ─────────────────────────────────────────────────────────────────────────────
#  Train or Load Classification Models
# ─────────────────────────────────────────────────────────────────────────────
def load_or_train_classification_bundle():
    """
    Load saved classification bundle or train classifiers on term-frequency features.
    """
    if os.path.exists(CLASSIFIER_BUNDLE_PATH):
        with open(CLASSIFIER_BUNDLE_PATH, "rb") as f:
            return pickle.load(f)

    abstracts, labels = load_corpus_data()
    y = labels.values

    vectorizer = CountVectorizer(max_features=200, stop_words="english")
    X_counts = vectorizer.fit_transform(abstracts)

    scaler = StandardScaler(with_mean=False).fit(X_counts)
    X_scaled = scaler.transform(X_counts)

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(probability=True, random_state=42),
    }

    trained_models = {}
    for name, clf in classifiers.items():
        clf.fit(X_scaled, y)
        trained_models[name] = clf

    bundle = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "models": trained_models,
    }
    with open(CLASSIFIER_BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return bundle

# ─────────────────────────────────────────────────────────────────────────────
#  Analyze Single Abstract
# ─────────────────────────────────────────────────────────────────────────────
def analyze_user_abstract(user_text, bundle):
    """
    Transform user_text into term-frequency features and predict with each classifier.
    Returns a dict of predictions.
    """
    vectorizer = bundle["vectorizer"]
    scaler = bundle["scaler"]
    models = bundle["models"]

    X_user_counts = vectorizer.transform([user_text])
    X_user_scaled = scaler.transform(X_user_counts)

    predictions = {}
    for name, clf in models.items():
        predictions[name] = clf.predict(X_user_scaled)[0]
    return predictions

# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Transportation Research Journal Recommender", layout="wide")

    # Custom CSS for styling, colored titles, and adjusted font sizes
    st.markdown(
        """
        <style>
        .page-title { text-align: center; font-size: 28px; margin-bottom: 20px; }
        .step-title { font-size: 20px; font-weight: 600; margin-bottom: 5px; }
        .step-desc { font-size: 16px; line-height: 1.5; margin-bottom: 20px; }
        .step-title.intro { color: #1f77b4; }
        .step-title.how { color: #ff7f0e; }
        .step-title.start { color: #2ca02c; }
        .result-box { border-left: 4px solid #2e6c80; padding: 10px; margin-top: 20px; }
        .survey { font-size: 16px; }
        .survey h2 { font-size: 20px; margin-bottom: 10px; }
        </style>
        """, unsafe_allow_html=True
    )

    # App Header with reduced font size
    st.markdown(
        "<div class='page-title'>📑 Transportation Research Journal Recommender</div>",
        unsafe_allow_html=True
    )

    # Step-by-step Introduction as rows
    with st.container():
        st.markdown("<div class='step-title intro'>1. Introduction</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='step-desc'>Have you wondered which part of the Transportation Research (TR) journal series is most relevant to your next manuscript? Do you feel overwhelmed by the breadth of TR journals? Try our TR Journal Recommender!</div>",
            unsafe_allow_html=True
        )

    with st.container():
        st.markdown("<div class='step-title how'>2. How It Works</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='step-desc'>This tool is based on a majority-vote ensemble of three machine learning classifiers—logistic regression, random forest, and support vector machine. Based on Web of Science data, the classification accuracy is approximately 65%.</div>",
            unsafe_allow_html=True
        )

    with st.container():
        st.markdown("<div class='step-title start'>3. Get Started</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='step-desc'>Paste your abstract below and click <strong>Analyze</strong> to receive a recommended TR journal.</div>",
            unsafe_allow_html=True
        )

    # Abstract Input Area
    input_col, _, _ = st.columns([3, 1, 3])
    with input_col:
        user_input = st.text_area("✍️ Paste your abstract here:", height=200)
        if st.button("Analyze Abstract"):
            if not user_input.strip():
                st.warning("Please enter an abstract to analyze.")
            else:
                with st.spinner("Processing..."):
                    bundle = load_or_train_classification_bundle()
                    preds = analyze_user_abstract(user_input, bundle)
                    from collections import Counter
                    vote_counts = Counter(preds.values())
                    recommended = vote_counts.most_common(1)[0][0]
                st.markdown(
                    f"<div class='result-box'><div class='step-title'>Recommended Journal</div><div class='step-desc'><strong>{recommended}</strong></div></div>",
                    unsafe_allow_html=True
                )
                st.success("Analysis complete!")

    # Survey Section
    st.markdown("---")
    st.markdown(
        "<div class='survey'><h2>🧠 vs 🤖 Challenge</h2><p>Do you feel you can beat the machine classifier? Please try! Test yourself by classifying five easy abstracts and five challenging ones, then compare your accuracy with the model’s performance.</p><p><a href='https://fsu.qualtrics.com/jfe/form/SV_81v6JJ7hXVd3eqq' target='_blank'>Take the survey</a></p></div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
