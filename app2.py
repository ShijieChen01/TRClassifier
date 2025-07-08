
import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
VECTORIZER_PATH = "vectorizer.pkl"
SCALER_PATH       = "scaler.pkl"
MODEL_PATHS = {
    #"LogisticRegression": "logistic_regression.pkl",
    #"RandomForest":       "random_forest.pkl",
    "SVC":                "svc.pkl",
}

CORPUS_PATH = "WOS_data.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
#  Load Raw Data
# ─────────────────────────────────────────────────────────────────────────────
def load_corpus_data():
    data = pd.read_excel(CORPUS_PATH, converters={"Abstract": str})
    abstracts = data["Abstract"].fillna("").tolist()
    labels    = data["Journal"].astype(str).reset_index(drop=True)
    return abstracts, labels

# ─────────────────────────────────────────────────────────────────────────────
#  Train or Load Classification Components
# ─────────────────────────────────────────────────────────────────────────────
def load_or_train_classification_bundle():
    # check if all files already exist
    all_exist = (
        os.path.exists(VECTORIZER_PATH)
        and os.path.exists(SCALER_PATH)
        and all(os.path.exists(p) for p in MODEL_PATHS.values())
    )
    if all_exist:
        # load each
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        trained_models = {}
        for name, path in MODEL_PATHS.items():
            with open(path, "rb") as f:
                trained_models[name] = pickle.load(f)
        return {"vectorizer": vectorizer, "scaler": scaler, "models": trained_models}

    # otherwise, train from scratch
    abstracts, labels = load_corpus_data()
    y = labels.values

    vectorizer = CountVectorizer(max_features=2000, stop_words="english")
    X_counts   = vectorizer.fit_transform(abstracts)

    scaler    = StandardScaler(with_mean=False).fit(X_counts)
    X_scaled  = scaler.transform(X_counts)

    classifiers = {
       # "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        #"RandomForest":       RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42),
        "SVC":                SVC(probability=True, random_state=42),
    }

    trained_models = {}
    # create a placeholder for status messages
    status_placeholder = st.empty()
    for name, clf in classifiers.items():
        status_placeholder.info(f"Training {name}…")
        clf.fit(X_scaled, y)
        trained_models[name] = clf
    # clear the status messages once done
    status_placeholder.empty()

    # save each component separately
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    for name, model in trained_models.items():
        with open(MODEL_PATHS[name], "wb") as f:
            pickle.dump(model, f)

    return {"vectorizer": vectorizer, "scaler": scaler, "models": trained_models}


# ─────────────────────────────────────────────────────────────────────────────
#  Analyze Single Abstract
# ─────────────────────────────────────────────────────────────────────────────
def analyze_user_abstract(user_text, bundle):
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

    # Custom CSS for styling
    st.markdown("""
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
    """, unsafe_allow_html=True)

    st.markdown("<div class='page-title'>Transportation Research Journal Recommender</div>", unsafe_allow_html=True)

    # Introduction steps...
    with st.container():
        st.markdown("<div class='step-title intro'>1. Introduction</div>", unsafe_allow_html=True)
        st.markdown("<div class='step-desc'>Have you wondered which part of the TR journal series is most relevant to your manuscript? Try our TR Journal Recommender!</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='step-title how'>2. How It Works</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='step-desc'>
        Built on a majority‐vote ensemble of logistic regression, random forest, and SVM. Trained on <strong>16,341</strong> abstracts (2010–2024) with baseline accuracy <strong>0.67</strong>. 
        <a href="https://github.com/ShijieChen01/TRClassifier" target="_blank">Source on GitHub</a>.
        </div>
        """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='step-title start'>3. Get Started</div>", unsafe_allow_html=True)
        st.markdown("<div class='step-desc'>Paste your abstract below and click <strong>Analyze</strong>.</div>", unsafe_allow_html=True)

    col_input, col_output = st.columns([3, 2])
    with col_input:
        user_input = st.text_area("Paste your abstract here:", height=200)
        analyze_clicked = st.button("Analyze Abstract")
        if analyze_clicked and not user_input.strip():
            st.warning("Please enter an abstract to analyze.")

    if analyze_clicked and user_input.strip():
        # this will show per-model status messages if training is needed
        bundle = load_or_train_classification_bundle()

        # now classify the user's abstract
        with st.spinner("Analyzing your abstract…"):
            preds = analyze_user_abstract(user_input, bundle)
            values = list(preds.values())
            unique_vals = set(values)
            if len(unique_vals) == 3:
                recommended = preds["SVC"]
            else:
                recommended = Counter(values).most_common(1)[0][0]

        with col_output:
            st.markdown(
                f"<div class='result-box'><div class='step-title'>Recommended Journal</div><div class='step-desc'><strong>{recommended}</strong></div></div>",
                unsafe_allow_html=True)
            st.success("Analysis complete!")

    # Survey Section
    st.markdown("---")
    st.markdown("""
    <div class='survey'>
      <h2>Classification Challenge</h2>
      <p>Beat the machine by classifying some abstracts yourself, then compare your accuracy!</p>
      <p><a href='https://fsu.qualtrics.com/jfe/form/SV_81v6JJ7hXVd3eqq' target='_blank'>Take the survey</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
