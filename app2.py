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
    Returns a dict with:
      - "vectorizer": CountVectorizer
      - "scaler": StandardScaler
      - "models": {
            "LogisticRegression": trained model,
            "RandomForest": trained model,
            "SVC": trained model
        }
    """
    if os.path.exists(CLASSIFIER_BUNDLE_PATH):
        with open(CLASSIFIER_BUNDLE_PATH, "rb") as f:
            return pickle.load(f)

    # Load data
    abstracts, labels = load_corpus_data()
    y = labels.values

    # Feature extraction
    vectorizer = CountVectorizer(max_features=200, stop_words="english")
    X_counts = vectorizer.fit_transform(abstracts)

    # Scaling
    scaler = StandardScaler(with_mean=False).fit(X_counts)
    X_scaled = scaler.transform(X_counts)

    # Define classifiers
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(probability=True, random_state=42),
    }

    # Train models
    trained_models = {}
    for name, clf in classifiers.items():
        clf.fit(X_scaled, y)
        trained_models[name] = clf

    # Bundle and save
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
    st.set_page_config(page_title="Abstract Classification", layout="wide")
    st.title("📑 Abstract Classification")
    st.markdown(
        """
        Paste an abstract below and click Analyze. The app will classify
        the abstract's journal using three classifiers:
        Logistic Regression, Random Forest, and SVC.
        """
    )

    user_input = st.text_area("Paste your abstract here:", height=200)

    if st.button("Analyze Abstract"):
        if not user_input.strip():
            st.warning("Please enter an abstract to analyze.")
            return

        with st.spinner("Processing..."):
            bundle = load_or_train_classification_bundle()
            preds = analyze_user_abstract(user_input, bundle)

        st.subheader("Journal Classification Predictions")
        results_df = pd.DataFrame(
            [
                {"Model": model_name, "Predicted Journal": label}
                for model_name, label in preds.items()
            ]
        )
        st.table(results_df)
        st.success("Analysis complete!")

if __name__ == "__main__":
    main()
