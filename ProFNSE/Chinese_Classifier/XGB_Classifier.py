import numpy as np
import pandas as pd
import jieba
import os
import time  # Import time module for timing
from xgboost import XGBClassifier  # Import XGBoost classifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)
from gensim.models import KeyedVectors


class Config:
    TOTAL_SAMPLES = 2541
    SKIP_ROWS = list(range(1, 141))
    TRAIN_RATIO = 0.8
    STOPWORDS_PATH = 'stop_words1.txt'
    INITIAL_BELIEVERS = 2
    RANDOM_STATE = 42
    VECTOR_CACHE_PATH = r'cached_vectors.npy'  # Word vector cache path
    WORD2VEC_PATH = r"sgns.weibo.word"


def load_word2vec():
    """Safely load word vectors"""
    try:
        return KeyedVectors.load_word2vec_format(Config.WORD2VEC_PATH, binary=True)
    except:
        try:
            return KeyedVectors.load_word2vec_format(Config.WORD2VEC_PATH, binary=False)
        except Exception as e:
            print(f"Word vector loading failed! Error details: {str(e)}")
            exit()


def preprocess_text(text, stopwords):
    """Text preprocessing function"""
    words = jieba.lcut(str(text))
    return [word for word in words if word not in stopwords and len(word) > 1]


def text_to_vector(words, word_vectors, dim=300):
    """Convert text to vector representation"""
    valid_words = [word for word in words if word in word_vectors]
    return np.mean([word_vectors[word] for word in valid_words], axis=0) if valid_words else np.zeros(dim)


def calculate_slope_and_final_believers(propagation_data, initial_believers=2):
    """Calculate propagation with correct initial believer count"""
    valid_indices = []
    slopes = []
    final_believers = []

    # Reconstruct propagation series (including day 0)
    full_series = []
    for idx, series in enumerate(propagation_data):
        try:
            # Insert initial value as day 0
            adjusted = np.insert(series.astype(float), 0, initial_believers)
            full_series.append(adjusted)
            valid_indices.append(idx)
        except:
            continue

    if len(full_series) == 0:
        return np.array([]), np.array([]), []

    full_series = np.array(full_series)
    print(f"Reconstructed propagation series dimension: {full_series.shape}")

    # Calculate daily changes (days 0-4)
    daily_changes = np.diff(full_series, axis=1)

    # Variance-driven weight calculation (based on change amounts)
    daily_variances = np.nanvar(daily_changes, axis=0, ddof=1)
    weights = daily_variances / daily_variances.sum()
    print(f"Weights based on variance (from {len(full_series)} samples): {np.round(weights, 2)}")

    # Time dimension (days 1-4)
    days = np.array([1, 2, 3, 4]).reshape(-1, 1)

    for idx, changes in enumerate(daily_changes):
        try:
            # Validate data integrity
            if len(changes) != 4 or np.isnan(changes).any():
                continue

            # Weighted linear regression
            model = LinearRegression()
            model.fit(days, changes.reshape(-1, 1), sample_weight=weights)

            slopes.append(model.coef_[0][0])
            final_believers.append(full_series[idx][-1])  # Use reconstructed final value
        except Exception as e:
            print(f"Calculation error at index {valid_indices[idx]}: {str(e)}")
            continue

    print(f"Valid samples: {len(slopes)}/{len(propagation_data)}")
    return np.array(slopes), np.array(final_believers), valid_indices[:len(slopes)]


def load_data(filepath):
    """Enhanced data loading"""
    df = pd.read_excel(filepath, nrows=Config.TOTAL_SAMPLES)

    required_columns = [
        'title', 'label', 'first day','second day', 'third day', 'fourth day',
        'StopWords Count', 'Sentiment analysis', 'StopWords Percentage', 'Noun Count',
        'Average word length', 'Readability', 'Characters', 'Words', 'Noun Phrases',
        'Sentences', 'Average characters per word', 'Average words per sentence', 'Average punctuations per sentence',
        'Modal verbs', 'Generalizing terms', 'Numbers_quantifiers', 'Positive words',
        'Negative words', 'Anxiety words', 'Exclamation marks', 'Sentiment polarity',
        'Subjective verbs', 'Unique words', 'Unique nouns', 'Unique_verbs', 'Unique_adjs',
        'Unique_advs', 'Typos', 'Hashtags', 'URLs', 'Question marks', 'Mentions', 'anger_score'
    ]

    propagation_data = df[['first day', 'second day', 'third day', 'fourth day']].values
    slopes, finals, valid_idx = calculate_slope_and_final_believers(propagation_data)

    df = df.iloc[valid_idx].copy()
    df['slope'] = slopes
    df['depth'] = finals
    return df[required_columns + ['slope', 'depth']], df['label']


def build_feature_pipeline(numeric_features):
    """Updated feature processor (adapted for XGBoost)"""
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())  # Standardization
    ])

    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=open(Config.STOPWORDS_PATH, encoding='utf-8').read().splitlines()
        ))
    ])

    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('txt', text_transformer, 'title')
    ])


def train_and_evaluate(X_train, X_test, y_train, y_test, numeric_features):
    """Model training and evaluation (XGBoost version)"""
    model = Pipeline([
        ('preprocessor', build_feature_pipeline(numeric_features)),
        ('classifier', XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    # Record training start time
    train_start = time.time()

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Calculate training time
    train_time = time.time() - train_start
    print(f"Model training completed, duration: {train_time:.2f} seconds")

    # Record prediction start time
    predict_start = time.time()

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate average prediction time per sample
    num_samples = len(X_test)
    predict_time = (time.time() - predict_start) / num_samples
    print(f"Average prediction time per sample: {predict_time * 1000:.4f} milliseconds")

    # Get prediction probabilities (for AUPRC calculation)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate core metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'auprc': average_precision_score(y_test, y_proba) if y_proba is not None else float('nan'),
        'train_time': train_time,
        'prediction_time_per_sample': predict_time
    }

    # Format output
    print("\n" + "=" * 40 + " Enhanced Evaluation Metrics " + "=" * 40)
    print(f"Accuracy:                   {metrics['accuracy']:.4f}")
    print(f"Precision (Weighted):       {metrics['precision']:.4f}")
    print(f"Recall (Weighted):          {metrics['recall']:.4f}")
    print(f"F1-Score (Weighted):        {metrics['f1_weighted']:.4f}")
    print(f"F1-Score (Macro):           {metrics['f1_macro']:.4f}")
    if not np.isnan(metrics['auprc']):
        print(f"AUPRC:                      {metrics['auprc']:.4f}")
    else:
        print("AUPRC:                      Not calculable (missing probability predictions)")
    print(f"Training time:              {metrics['train_time']:.2f} seconds")
    print(f"Prediction time per sample: {metrics['prediction_time_per_sample'] * 1000:.4f} milliseconds")

    return metrics


if __name__ == "__main__":
    wv = load_word2vec()
    print("Word vectors loaded successfully! Vocabulary size:", len(wv.key_to_index))
    file_path = r""

    try:
        print("Loading data...")
        features, labels = load_data(file_path)

        # ------------- Improved word vector processing -------------
        print("\nProcessing word vector features...")
        stopwords = open(Config.STOPWORDS_PATH, encoding='utf-8').read().splitlines()

        # Try loading cached vectors
        if os.path.exists(Config.VECTOR_CACHE_PATH):
            print("Detected word vector cache, loading directly...")
            vectors = np.load(Config.VECTOR_CACHE_PATH)
        else:
            print("No cache found, generating word vectors and saving...")
            # Text preprocessing
            features['segmented_title'] = features['title'].apply(
                lambda x: preprocess_text(x, stopwords)
            )
            # Generate word vectors
            dim = wv.vector_size
            vectors = np.vstack(features['segmented_title'].apply(
                lambda x: text_to_vector(x, wv, dim)
            ))
            # Save cache
            np.save(Config.VECTOR_CACHE_PATH, vectors)
            print(f"Word vectors cached at: {Config.VECTOR_CACHE_PATH}")

        # Add word vectors to features
        vector_columns = [f'vec_{i}' for i in range(wv.vector_size)]
        features[vector_columns] = vectors

        base_numeric_features = [
            'Words','Unique words','Average characters per word','Subjective verbs','Readability','Sentences','Numbers_quantifiers',
            'Hashtags','Modal verbs','URLs','Typos','Exclamation marks','Positive words','Sentiment polarity','slope'
        ]
        numeric_features = base_numeric_features + vector_columns

        print("\nSplitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            train_size=Config.TRAIN_RATIO,
            stratify=labels,
            random_state=Config.RANDOM_STATE
        )

        print("\nFeature dimensions:")
        print(f"Numeric features: {len(numeric_features)} dimensions ({len(vector_columns)} word vectors included)")
        print(f"Text features: TF-IDF (5000 dimensions)")

        results = train_and_evaluate(X_train, X_test, y_train, y_test, numeric_features)

        # Final output
        print("\nFinal evaluation results:")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Precision: {results['precision']:.2%}")
        print(f"Recall: {results['recall']:.2%}")
        print(f"F1-Score (Weighted): {results['f1_weighted']:.2%}")
        print(f"F1-Score (Macro):    {results['f1_macro']:.4f}")
        print(f"AUPRC:       {results['auprc']:.4f}")
        print(f"Training time:    {results['train_time']:.2f} seconds")
        print(f"Prediction time per sample: {results['prediction_time_per_sample'] * 1000:.4f} milliseconds")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")