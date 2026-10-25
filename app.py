from pathlib import Path
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Food Review Sentiment Analysis", page_icon="🍽️", layout="wide")
st.title("🍽️ Food Review Sentiment Analysis System")
st.write("Analyze food reviews, explore sentiment trends, and download predictions.")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "zom_rev_with_sentiment(1).csv"
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"
LABELS = ["bad", "neutral", "good"]

@st.cache_data
def load_dataset(file_bytes=None):
    from io import BytesIO
    if file_bytes is not None:
        df = pd.read_csv(BytesIO(file_bytes))
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]
    if "review" not in df.columns:
        candidates = [c for c in df.columns if "review" in c or "text" in c or "comment" in c]
        if candidates:
            df = df.rename(columns={candidates[0]: "review"})
    if "review" not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame()

    label_map = {
        "positive": "good", "good": "good", "positive review": "good",
        "negative": "bad", "bad": "bad", "negative review": "bad",
        "neutral": "neutral", "average": "neutral", "mixed": "neutral"
    }
    df["review"] = df["review"].fillna("").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.strip().str.lower().map(label_map)
    df = df[df["review"].ne("") & df["sentiment"].isin(LABELS)].copy()
    return df.drop_duplicates(subset=["review", "sentiment"])

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

uploaded = st.sidebar.file_uploader("Optional: upload a CSV dataset", type=["csv"])
df = load_dataset(uploaded.getvalue() if uploaded else None)

if df.empty:
    st.error(
        "Dataset not loaded or missing required columns. Keep the project's CSV beside app.py "
        "with `review` and `sentiment` columns."
    )
    st.stop()

model = load_model()
if model is None:
    st.error("Trained model not found. First run `py train_model.py` in PowerShell, then restart this app.")
    st.code("py train_model.py\npy -m streamlit run app.py", language="powershell")
    st.stop()

counts = df["sentiment"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total reviews", f"{len(df):,}")
c2.metric("Good", f"{int(counts.get('good', 0)):,}")
c3.metric("Bad", f"{int(counts.get('bad', 0)):,}")
c4.metric("Neutral", f"{int(counts.get('neutral', 0)):,}")

overview, predict, batch, data_tab = st.tabs(["📊 Overview", "🔎 Analyze a Review", "📁 Batch Analysis", "🗂️ Dataset"])

with overview:
    left, right = st.columns(2)
    with left:
        sentiment_counts = counts.reindex(LABELS, fill_value=0)
        fig, ax = plt.subplots()
        ax.bar(["Bad", "Neutral", "Good"], sentiment_counts.values)
        ax.set_ylabel("Number of reviews")
        ax.set_title("Sentiment distribution")
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        if "rating" in df.columns:
            rating_counts = pd.to_numeric(df["rating"], errors="coerce").value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.bar(rating_counts.index.astype(str), rating_counts.values)
            ax.set_xlabel("Rating")
            ax.set_ylabel("Number of reviews")
            ax.set_title("Rating distribution")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No rating column found.")
    st.dataframe(df.head(10), use_container_width=True)

with predict:
    review_text = st.text_area("Enter a food review", placeholder="The food was delicious and the service was excellent.")
    if st.button("Analyze sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Enter a review first.")
        else:
            pred = model.predict([review_text.strip()])[0]
            proba = model.predict_proba([review_text.strip()])[0]
            classes = model.named_steps["classifier"].classes_
            if pred == "good":
                st.success("Predicted sentiment: GOOD 😊")
            elif pred == "bad":
                st.error("Predicted sentiment: BAD 😞")
            else:
                st.info("Predicted sentiment: NEUTRAL 😐")
            prob_df = pd.DataFrame({
                "Sentiment": [x.title() for x in classes],
                "Estimated probability": proba
            }).sort_values("Estimated probability", ascending=False)
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

with batch:
    st.write("Upload a CSV containing a `review` column (or `text`/`comment`).")
    batch_file = st.file_uploader("Upload reviews CSV", type=["csv"], key="batch")
    if batch_file:
        batch_df = pd.read_csv(batch_file)
        batch_df.columns = [str(c).strip().lower() for c in batch_df.columns]
        if "review" not in batch_df.columns:
            candidates = [c for c in batch_df.columns if "review" in c or c in ["text", "comment"]]
            if candidates:
                batch_df = batch_df.rename(columns={candidates[0]: "review"})
        if "review" not in batch_df.columns:
            st.error("CSV needs a review/text/comment column.")
        else:
            batch_df["review"] = batch_df["review"].fillna("").astype(str)
            batch_df = batch_df[batch_df["review"].str.strip().ne("")].copy()
            batch_df["predicted_sentiment"] = model.predict(batch_df["review"].tolist()).astype(str)
            batch_df["confidence"] = model.predict_proba(batch_df["review"].tolist()).max(axis=1).round(4)
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "Download predictions CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="food_review_sentiment_predictions.csv",
                mime="text/csv"
            )

with data_tab:
    filter_labels = st.multiselect("Filter sentiment", LABELS, default=LABELS)
    search = st.text_input("Search review text")
    filtered = df[df["sentiment"].isin(filter_labels)]
    if search:
        filtered = filtered[filtered["review"].str.contains(re.escape(search), case=False, na=False)]
    st.write(f"{len(filtered):,} matching reviews")
    st.dataframe(filtered.head(2000), use_container_width=True)
    st.download_button(
        "Download filtered dataset",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_food_reviews.csv",
        mime="text/csv"
    )

st.caption("Educational project. Model predictions and probability estimates are not guaranteed to be correct.")
