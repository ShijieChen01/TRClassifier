import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import pickle
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

st.set_page_config(page_title="Abstract Topic & Classification", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
#  Constants / File paths
# ─────────────────────────────────────────────────────────────────────────────
TOPIC_BUNDLE_PATH = "bertopic_bundle.pkl"
CLASSIFIER_BUNDLE_PATH = "classification_bundle.pkl"
CORPUS_PATH = "WOS_data.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
#  Utility: Load raw data
# ─────────────────────────────────────────────────────────────────────────────
# @st.cache(allow_output_mutation=True)
def load_corpus_data():
    """
    Load abstracts, authors, and labels from Excel.
    """
    data = pd.read_excel(CORPUS_PATH, converters={"Abstract": str, "Authors": str})
    abstracts = data["Abstract"].fillna("").tolist()
    authors = data["Authors"].fillna("").tolist()
    labels = data["Journal"].astype(str).reset_index(drop=True)
    return abstracts, authors, labels

# ─────────────────────────────────────────────────────────────────────────────
#  Text Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_texts_corpus(texts, authors_list):
    """
    Preprocess a list of documents for topic modeling:
      - Lowercase, remove punctuation & digits
      - Tokenize, remove stopwords + custom stopwords
      - Lemmatize
      - Remove tokens appearing in <5% or >95% of documents
    Returns List[List[str]] of tokens per document.
    """
    stop_words = set(stopwords.words("english"))
    custom = {
        "problem","result","system","research","solution","ltd","elsevier",
        "study","paper","right","rights","reserved","using","propose","proposed",
        "transportation","transport","use","according","aim","also"
    }
    stop_words |= custom

    cleaned_tokens = []
    for doc in texts:
        no_punct = re.sub(r"[^\w\s]", " ", doc)
        no_digits = re.sub(r"\d+", " ", no_punct)
        tokens = word_tokenize(no_digits.lower())
        cleaned_tokens.append(tokens)

    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    for tokens in cleaned_tokens:
        filtered = [lemmatizer.lemmatize(tok) for tok in tokens
                    if tok not in stop_words and len(tok) > 1]
        lemmatized.append(filtered)

    # Compute document frequencies
    num_docs = len(lemmatized)
    freq_counter = Counter()
    for tokens in lemmatized:
        freq_counter.update(set(tokens))

    low_thresh = 0.05 * num_docs
    high_thresh = 0.95 * num_docs

    filtered_corpus = []
    for tokens in lemmatized:
        filtered = [tok for tok in tokens
                    if low_thresh <= freq_counter[tok] <= high_thresh]
        filtered_corpus.append(filtered)

    return filtered_corpus

def preprocess_single_text(text):
    """
    Preprocess a single document:
      - Lowercase, remove punctuation & digits
      - Tokenize, remove stopwords + custom stopwords
      - Lemmatize
    Returns List[str] of tokens.
    """
    stop_words = set(stopwords.words("english"))
    custom = {
        "problem","result","system","research","solution","ltd","elsevier",
        "study","paper","right","rights","reserved","using","propose","proposed",
        "transportation","transport","use","according","aim","also"
    }
    stop_words |= custom

    no_punct = re.sub(r"[^\w\s]", " ", text)
    no_digits = re.sub(r"\d+", " ", no_punct)
    tokens = word_tokenize(no_digits.lower())

    lemmatizer = WordNetLemmatizer()
    filtered = [lemmatizer.lemmatize(tok) for tok in tokens
                if tok not in stop_words and len(tok) > 1]
    return filtered

# ─────────────────────────────────────────────────────────────────────────────
#  Topic Model: train or load
# ─────────────────────────────────────────────────────────────────────────────
def load_or_train_topic_bundle():
    """
    If TOPIC_BUNDLE_PATH exists, load and return its contents.
    Otherwise, train BERTopic on the corpus, build CountVectorizer,
    save everything to TOPIC_BUNDLE_PATH, and return.
    Returns:
      topic_model: trained BERTopic instance
      topic_probs_df: DataFrame (n_docs × n_topics)
      topic_dist_array: np.ndarray of shape (n_docs, n_topics)
      vectorizer: fitted CountVectorizer
      df_term_doc: DataFrame (n_docs × n_terms)
      topic_col_names: List[str] of "Topic 0", "Topic 1", ...
      abstracts: List[str]
      labels: pd.Series
    """
    if os.path.exists(TOPIC_BUNDLE_PATH):
        with open(TOPIC_BUNDLE_PATH, "rb") as f:
            bundle = pickle.load(f)
        return (
            bundle["topic_model"],
            bundle["topic_probs_df"],
            bundle["topic_dist_array"],
            bundle["vectorizer"],
            bundle["df_term_doc"],
            bundle["topic_col_names"],
            bundle["abstracts"],
            bundle["labels"],
        )

    # Otherwise, train from scratch
    abstracts, authors, labels = load_corpus_data()
    tokenized = preprocess_texts_corpus(abstracts, authors)
    preprocessed_strs = [" ".join(tokens) for tokens in tokenized]

    # Set random seeds for reproducibility
    np.random.seed(42)

    # Step 1: Embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2: UMAP
    umap_model = UMAP(
        n_neighbors=15,
        n_components=100,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    # Step 3: HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    # Step 4: BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=50,
        verbose=False,
        calculate_probabilities=True,
        top_n_words=50,
        n_gram_range=(1, 2)
    )
    topics, probs = topic_model.fit_transform(preprocessed_strs)
    topic_dist_array = np.array(probs)  # shape: (n_docs, n_topics)
    n_topics = topic_dist_array.shape[1]
    topic_col_names = [f"Topic {i}" for i in range(n_topics)]
    topic_probs_df = pd.DataFrame(topic_dist_array, columns=topic_col_names)

    # Step 5: Term-document matrix
    vectorizer = CountVectorizer(max_features=2000)
    X_counts = vectorizer.fit_transform(preprocessed_strs)
    terms = vectorizer.get_feature_names_out()
    df_term_doc = pd.DataFrame(X_counts.toarray(), columns=terms)

    # Save everything
    bundle = {
        "topic_model": topic_model,
        "topic_probs_df": topic_probs_df,
        "topic_dist_array": topic_dist_array,
        "vectorizer": vectorizer,
        "df_term_doc": df_term_doc,
        "topic_col_names": topic_col_names,
        "abstracts": abstracts,
        "labels": labels,
    }
    with open(TOPIC_BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return (
        topic_model,
        topic_probs_df,
        topic_dist_array,
        vectorizer,
        df_term_doc,
        topic_col_names,
        abstracts,
        labels,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Classification Models: train or load
# ─────────────────────────────────────────────────────────────────────────────
def load_or_train_classification_bundle():
    """
    If CLASSIFIER_BUNDLE_PATH exists, load and return its contents.
    Otherwise:
      - Load or train topic bundle
      - Build feature matrices for 'topic' and 'topic_term'
      - Train three classifiers (LogisticRegression, RandomForest, SVC)
        on both feature sets
      - Save models + scalers to CLASSIFIER_BUNDLE_PATH
      - Return models_dict
    Returns:
      models_dict: {
        'topic': {
            'LogisticRegression': (clf, scaler_topic),
            'RandomForest': (clf, scaler_topic),
            'SVC': (clf, scaler_topic)
        },
        'topic_term': {
            'LogisticRegression': (clf, scaler_topic_term),
            'RandomForest': (clf, scaler_topic_term),
            'SVC': (clf, scaler_topic_term)
        }
      }
    """
    if os.path.exists(CLASSIFIER_BUNDLE_PATH):
        with open(CLASSIFIER_BUNDLE_PATH, "rb") as f:
            models_dict = pickle.load(f)
        return models_dict

    # Otherwise, train from scratch
    (
        topic_model,
        topic_probs_df,
        topic_dist_array,
        vectorizer,
        df_term_doc,
        topic_col_names,
        abstracts,
        labels,
    ) = load_or_train_topic_bundle()

    y = labels.values

    # Feature matrices
    X_topic = topic_dist_array  # shape: (n_docs, n_topics)
    X_term = df_term_doc.values  # shape: (n_docs, n_terms)
    X_topic_term = np.hstack((X_topic, X_term))  # shape: (n_docs, n_topics + n_terms)

    # Scaleers
    scaler_topic = StandardScaler().fit(X_topic)
    X_topic_scaled = scaler_topic.transform(X_topic)

    scaler_topic_term = StandardScaler().fit(X_topic_term)
    X_topic_term_scaled = scaler_topic_term.transform(X_topic_term)

    # Define classifiers with fixed seeds
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(probability=True, random_state=42),
    }

    models_dict = {"topic": {}, "topic_term": {}}

    # Train on topic-only
    for name, base_clf in classifiers.items():
        clf = base_clf.__class__(**base_clf.get_params())
        clf.fit(X_topic_scaled, y)
        models_dict["topic"][name] = (clf, scaler_topic)

    # Train on topic + term
    for name, base_clf in classifiers.items():
        clf = base_clf.__class__(**base_clf.get_params())
        clf.fit(X_topic_term_scaled, y)
        models_dict["topic_term"][name] = (clf, scaler_topic_term)

    # Save to disk
    with open(CLASSIFIER_BUNDLE_PATH, "wb") as f:
        pickle.dump(models_dict, f)

    return models_dict

# ─────────────────────────────────────────────────────────────────────────────
#  Analyze a single user-provided abstract
# ─────────────────────────────────────────────────────────────────────────────
def analyze_user_abstract(user_text):
    """
    Given a single abstract string:
      - Preprocess into tokens
      - Build word frequency DataFrame
      - Compute topic distribution via loaded topic_model
      - Run classification predictions for each of the six (feature set × model) combos
    Returns:
      {
        "tokens": List[str],
        "word_freq_df": DataFrame(["word", "frequency"]),
        "topic_distribution": DataFrame (1 × n_topics),
        "predictions": {
            "topic": {"LogisticRegression": label, "RandomForest": label, "SVC": label},
            "topic_term": {"LogisticRegression": label, "RandomForest": label, "SVC": label}
        }
      }
    """
    # 1) Preprocess user text
    tokens = preprocess_single_text(user_text)
    preprocessed_str = " ".join(tokens)

    # 2) Word frequency
    freq_counter = Counter(tokens)
    word_freq_df = pd.DataFrame(
        freq_counter.most_common(), columns=["word", "frequency"]
    )

    # 3) Load models
    (
        topic_model,
        topic_probs_df,
        topic_dist_array,
        vectorizer,
        df_term_doc,
        topic_col_names,
        abstracts,
        labels,
    ) = load_or_train_topic_bundle()
    models_dict = load_or_train_classification_bundle()

    # 4) Topic distribution for user abstract
    user_topics, user_probs = topic_model.transform([preprocessed_str])
    user_probs_array = np.array(user_probs)[0]  # shape: (n_topics,)
    user_topic_df = pd.DataFrame(
        user_probs_array.reshape(1, -1), columns=topic_col_names
    )

    # 5) Classification predictions
    predictions = {"topic": {}, "topic_term": {}}

    # Feature set "topic"
    X_user_topic = user_probs_array.reshape(1, -1)
    for model_name, (clf, scaler) in models_dict["topic"].items():
        X_scaled = scaler.transform(X_user_topic)
        pred_label = clf.predict(X_scaled)[0]
        predictions["topic"][model_name] = pred_label

    # Feature set "topic_term"
    X_user_term = vectorizer.transform([preprocessed_str]).toarray()  # shape: (1, n_terms)
    X_user_tt = np.hstack((X_user_topic, X_user_term))  # shape: (1, n_topics + n_terms)
    for model_name, (clf, scaler) in models_dict["topic_term"].items():
        X_scaled = scaler.transform(X_user_tt)
        pred_label = clf.predict(X_scaled)[0]
        predictions["topic_term"][model_name] = pred_label

    return {
        "tokens": tokens,
        "word_freq_df": word_freq_df,
        "topic_distribution": user_topic_df,
        "predictions": predictions,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📑 Abstract Topic Distribution & Classification")
    st.markdown(
        """
        Enter an abstract below. The app will:
        1. Preprocess the abstract and display word frequencies.
        2. Compute the topic distribution using a BERTopic model (or train if absent).
        3. Classify the abstract's journal using two feature sets (topic-only, topic+term frequency) 
           and three classifiers (Logistic Regression, Random Forest, SVC), loading them if they exist.
        """
    )

    user_input = st.text_area("Paste your abstract here:", height=200)

    if st.button("Analyze Abstract"):
        if not user_input.strip():
            st.warning("Please enter an abstract to analyze.")
            return

        with st.spinner("Processing..."):
            results = analyze_user_abstract(user_input)

        # Display word frequency (top 20)
        st.subheader("Word Frequency (Top 20)")
        wf_df = results["word_freq_df"].head(20)
        st.table(wf_df)

        # Display topic distribution
        st.subheader("Topic Distribution")
        td_df = results["topic_distribution"].T
        td_df.columns = ["Probability"]
        td_df["Topic"] = td_df.index
        td_df = td_df.reset_index(drop=True)[["Topic", "Probability"]]
        st.dataframe(td_df.style.format({"Probability": "{:.4f}"}))

        # Display classification predictions
        st.subheader("Journal Classification Predictions")
        pred_dict = results["predictions"]
        display_rows = []
        for feat_set, model_preds in pred_dict.items():
            for model_name, label in model_preds.items():
                display_rows.append(
                    {"Feature Set": feat_set, "Model": model_name, "Predicted Journal": label}
                )
        display_df = pd.DataFrame(display_rows)
        st.table(display_df)

        st.success("Analysis complete!")

if __name__ == "__main__":
    main()
