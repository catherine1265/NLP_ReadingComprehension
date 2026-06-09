import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from model_loader import ensure_models

def compute_tfidf_summary():
    ensure_models()
    df_summary = pd.read_csv('squad_preprocessed.csv').dropna(subset=['answer'])

    tfidf_vec    = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    tfidf_matrix = tfidf_vec.fit_transform(df_summary['answer'])

    feature_names = tfidf_vec.get_feature_names_out()
    mean_tfidf    = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_idx       = mean_tfidf.argsort()[::-1][:20]

    terms          = [feature_names[i] for i in top_idx]
    scores         = [float(mean_tfidf[i]) for i in top_idx]
    top_terms_str  = ', '.join(f'"{t}"' for t in terms[:5])

    return {
        'terms'        : terms,
        'scores'       : scores,
        'n_docs'       : tfidf_matrix.shape[0],
        'vocab_size'   : tfidf_matrix.shape[1],
        'top_terms_str': top_terms_str,
    }

def plot_tfidf(summary: dict):
    terms  = summary['terms']
    scores = summary['scores']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(terms[::-1], scores[::-1], color='steelblue', edgecolor='white')
    ax.set_title('Top 20 Terms by Mean TF-IDF Weight (Answers)')
    ax.set_xlabel('Mean TF-IDF Score')
    plt.tight_layout()
    return fig
