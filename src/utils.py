<<<<<<< HEAD
import os
import joblib

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_model_paths():
    root = get_project_root()
    model_path = os.path.join(root, "models", "fake_news_model.pkl")
    vectorizer_path = os.path.join(root, "models", "tfidf_vectorizer.pkl")
    return model_path, vectorizer_path

def load_model_and_vectorizer():
    model_path, vectorizer_path = get_model_paths()
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
=======
import os
import joblib

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_model_paths():
    root = get_project_root()
    model_path = os.path.join(root, "models", "fake_news_model.pkl")
    vectorizer_path = os.path.join(root, "models", "tfidf_vectorizer.pkl")
    return model_path, vectorizer_path

def load_model_and_vectorizer():
    model_path, vectorizer_path = get_model_paths()
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
>>>>>>> aa92bda (Added training & prediction scripts + repo polish)
