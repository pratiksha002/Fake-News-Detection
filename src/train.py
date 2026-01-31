<<<<<<< HEAD
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import clean_text

def main():
    # Paths
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_fake = os.path.join(root, "data", "Fake.csv")
    data_true = os.path.join(root, "data", "True.csv")
    model_dir = os.path.join(root, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Load
    fake_df = pd.read_csv(data_fake)
    true_df = pd.read_csv(data_true)

    # Labels
    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Use only text (recommended)
    df["content"] = df["text"].astype(str)
    df["clean_content"] = df["content"].apply(clean_text)

    # Remove empty
    df = df[df["clean_content"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)

    X = df["clean_content"]
    y = df["label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("✅ Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save
    joblib.dump(model, os.path.join(model_dir, "fake_news_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

    print("\n✅ Saved model + vectorizer inside models/")

if __name__ == "__main__":
    main()
=======
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import clean_text

def main():
    # Paths
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_fake = os.path.join(root, "data", "Fake.csv")
    data_true = os.path.join(root, "data", "True.csv")
    model_dir = os.path.join(root, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Load
    fake_df = pd.read_csv(data_fake)
    true_df = pd.read_csv(data_true)

    # Labels
    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Use only text (recommended)
    df["content"] = df["text"].astype(str)
    df["clean_content"] = df["content"].apply(clean_text)

    # Remove empty
    df = df[df["clean_content"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)

    X = df["clean_content"]
    y = df["label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("✅ Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save
    joblib.dump(model, os.path.join(model_dir, "fake_news_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

    print("\n✅ Saved model + vectorizer inside models/")

if __name__ == "__main__":
    main()
>>>>>>> aa92bda (Added training & prediction scripts + repo polish)
