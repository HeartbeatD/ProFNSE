import os
import time  # Import time module for timing
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve  # Metrics for AUPRC calculation
)
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords


# ==================== Configuration Section ====================
class Config:
    GLOVE_PATH = r"glove.twitter.27B.200d.txt"
    DATASET_PATH = r""
    VECTOR_DIM = 200
    CACHE_PATH = 'cached_vectors.npy'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    INITIAL_BELIEVERS = 2
    GLOVE_BIN_PATH = r"glove.twitter.27B.200d.gensim.bin"


# ==================== Initialize NLTK ====================
nltk.data.path.append(r"")  # Set to your actual path
try:
    STOPWORDS = frozenset(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
except LookupError:
    nltk.download(['punkt', 'wordnet', 'stopwords'], quiet=True)
    STOPWORDS = frozenset(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()


# ==================== Core Functions ====================
def calculate_slope_and_final_believers(propagation_data, initial_believers=2):
    """Calculate propagation slope and final believers with initial count"""
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

    # Calculate daily changes (days 0-4)
    daily_changes = np.diff(full_series, axis=1)

    # Variance-based weight calculation
    daily_variances = np.nanvar(daily_changes, axis=0, ddof=1)
    weights = daily_variances / daily_variances.sum()

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
            final_believers.append(full_series[idx][-1])
        except Exception as e:
            print(f"Index {valid_indices[idx]} calculation exception: {str(e)}")
            continue

    return np.array(slopes), np.array(final_believers), valid_indices[:len(slopes)]


def preprocess_text(text):
    """Text preprocessing function"""
    text = str(text).lower().strip()
    tokens = nltk.word_tokenize(text)
    processed = [
        LEMMATIZER.lemmatize(w)
        for w in tokens
        if w.isalpha() and w not in STOPWORDS and len(w) > 2
    ]
    return ' '.join(processed)


def load_glove_model():
    """Load GloVe model efficiently"""
    if os.path.exists(Config.GLOVE_BIN_PATH):
        return KeyedVectors.load(Config.GLOVE_BIN_PATH, mmap='r')

    w2v_file = Config.GLOVE_PATH + '.word2vec'
    if not os.path.exists(w2v_file):
        glove2word2vec(Config.GLOVE_PATH, w2v_file)

    model = KeyedVectors.load_word2vec_format(w2v_file)
    model.save(Config.GLOVE_BIN_PATH)
    return model


def text_to_vector(tokens, word_vectors):
    """Convert text tokens to vector representation"""
    valid_words = [w for w in tokens if w in word_vectors]
    if not valid_words:
        return np.zeros(Config.VECTOR_DIM)
    return np.mean([word_vectors[w] for w in valid_words], axis=0)


def load_data():
    """Load and preprocess dataset"""
    df = pd.read_excel(Config.DATASET_PATH)

    # Calculate propagation metrics
    propagation_data = df[['first day', 'second day', 'third day', 'fourth day']].values
    slopes, finals, valid_idx = calculate_slope_and_final_believers(propagation_data)

    # Filter valid indices
    df = df.iloc[valid_idx].copy()
    df['slope'] = slopes
    df['depth'] = finals

    # Text preprocessing
    df['processed_text'] = df['title'].apply(preprocess_text)

    # Load or generate word vectors
    if os.path.exists(Config.CACHE_PATH):
        vectors = np.load(Config.CACHE_PATH)
    else:
        wv = load_glove_model()
        vectors = np.vstack(df['processed_text'].apply(
            lambda x: text_to_vector(x.split(), wv)
        ))
        np.save(Config.CACHE_PATH, vectors)

    # Add vector columns to DataFrame
    vec_columns = [f'vec_{i}' for i in range(Config.VECTOR_DIM)]
    df = pd.concat([
        df,
        pd.DataFrame(vectors, columns=vec_columns, index=df.index)
    ], axis=1)

    return df


def build_feature_pipeline():
    """Build feature processing pipeline"""
    numeric_features = [
        'StopWords Count', 'Unique words', 'Average words per sentence',
        'Subjective verbs', 'Noun Count', 'Negative words', 'Readability',
        'Average word length', 'Sentiment polarity', 'slope'
    ]
    vec_columns = [f'vec_{i}' for i in range(Config.VECTOR_DIM)]

    return ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('glove', 'passthrough', vec_columns),
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        ), 'processed_text')
    ])


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics"""
    # Record prediction start time
    predict_start = time.time()

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate prediction time
    predict_time = time.time() - predict_start

    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'auprc': average_precision_score(y_test, y_proba) if y_proba is not None else float('nan'),
        'prediction_time': predict_time
    }

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba) if y_proba is not None else (None, None, None)

    return metrics, precision, recall


# ==================== Main Program ====================
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('label', axis=1),
        df['label'],
        test_size=Config.TEST_SIZE,
        stratify=df['label'],
        random_state=Config.RANDOM_STATE
    )

    print("Building model...")
    model = Pipeline([
        ('features', build_feature_pipeline()),
        ('classifier', LogisticRegression(
            penalty='l2',  # L2 regularization
            C=1.0,  # Regularization strength
            class_weight='balanced',  # Handle class imbalance
            solver='liblinear',  # Suitable for small datasets
            max_iter=1000,  # Prevent non-convergence
            random_state=Config.RANDOM_STATE,
            verbose=1  # Show training logs
        ))
    ])

    print("Training model...")
    start_train_time = time.time()  # Record training start time
    model.fit(X_train, y_train)
    train_time = time.time() - start_train_time  # Calculate training time

    # Evaluate model performance
    metrics, precision, recall = evaluate_model(model, X_test, y_test)

    # Calculate per-sample prediction time
    start_pred_time = time.time()
    for _ in range(len(X_test)):
        model.predict(X_test.iloc[[0]])  # Predict single sample
    avg_pred_time = (time.time() - start_pred_time) / len(X_test)

    # Format output
    print("\n" + "=" * 50 + " Comprehensive Evaluation Results " + "=" * 50)
    print(f"Training time:                     {train_time:.4f} seconds")
    print(f"Per-sample prediction time:        {avg_pred_time * 1000:.4f} milliseconds")
    print(f"Batch prediction time:             {metrics['prediction_time']:.4f} seconds")
    print("-" * 100)
    print(f"Accuracy:                          {metrics['accuracy']:.4f}")
    print(f"Precision (Weighted):             {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted):                {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (Weighted):              {metrics['f1_weighted']:.4f}")
    print(f"F1-Score (Macro):                 {metrics['f1_macro']:.4f}")
    print(f"AUPRC:                            {metrics['auprc']:.4f} (Area Under the Precision-Recall Curve)")