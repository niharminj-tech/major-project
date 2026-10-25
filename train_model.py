from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "zom_rev_with_sentiment(1).csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"

LABELS = ["bad", "neutral", "good"]

def prepare_data(df):
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Find a likely review-text column if it is not named exactly "review".
    if "review" not in df.columns:
        candidates = [c for c in df.columns if "review" in c or "text" in c or "comment" in c]
        if not candidates:
            raise ValueError(
                "Could not find a review text column. Rename the text column to 'review'. "
                f"Available columns: {list(df.columns)}"
            )
        df = df.rename(columns={candidates[0]: "review"})

    # If sentiment is missing, derive labels from a numeric rating where possible.
    if "sentiment" not in df.columns:
        if "rating" not in df.columns:
            raise ValueError(
                "Dataset must contain a 'sentiment' column or a numeric 'rating' column."
            )
        rating = pd.to_numeric(df["rating"], errors="coerce")
        df["sentiment"] = rating.apply(
            lambda x: "good" if x >= 4 else ("bad" if x <= 2 else ("neutral" if x == 3 else None))
        )

    df["review"] = df["review"].fillna("").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.strip().str.lower()

    # Standardize common label variants.
    label_map = {
        "positive": "good", "good": "good", "positive review": "good",
        "negative": "bad", "bad": "bad", "negative review": "bad",
        "neutral": "neutral", "average": "neutral", "mixed": "neutral"
    }
    df["sentiment"] = df["sentiment"].map(label_map)
    df = df[df["review"].ne("") & df["sentiment"].isin(LABELS)].copy()
    df = df.drop_duplicates(subset=["review", "sentiment"])

    if df.empty:
        raise ValueError(
            "No usable rows found. Check that the dataset has review text and sentiment labels "
            "Good/Bad/Neutral (or Positive/Negative/Neutral)."
        )
    if df["sentiment"].nunique() < 2:
        raise ValueError("At least two sentiment classes are needed to train a classifier.")
    if df["sentiment"].value_counts().min() < 2:
        raise ValueError(
            "Each sentiment class needs at least 2 examples for a stratified train/test split. "
            "Add more labeled examples or adjust the split."
        )
    return df

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place zom_rev_with_sentiment(1).csv in the same folder as train_model.py."
        )

    print("Loading dataset...")
    df = prepare_data(pd.read_csv(DATA_PATH))
    print(f"Usable reviews: {len(df):,}")
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100000,
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    print("\nTraining TF-IDF + Logistic Regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; order: bad, neutral, good):")
    print(confusion_matrix(y_test, predictions, labels=LABELS))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")
    print("Training complete. Run the Streamlit app after this script finishes.")

if __name__ == "__main__":
    main()
