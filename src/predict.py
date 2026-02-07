import sys
from preprocessing import clean_text
from utils import load_model_and_vectorizer

def predict_news(text):
    model, vectorizer = load_model_and_vectorizer()
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    proba = model.predict_proba(vec)[0]
    fake_prob = float(proba[0])
    real_prob = float(proba[1])

    diff = abs(fake_prob - real_prob)

    if diff < 0.10:
        label = "UNCERTAIN ⚠️"
        confidence = max(fake_prob, real_prob)
    elif real_prob > fake_prob:
        label = "REAL ✅"
        confidence = real_prob
    else:
        label = "FAKE ❌"
        confidence = fake_prob

    return label, confidence, fake_prob, real_prob

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"news text here\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    label, confidence, fake_prob, real_prob = predict_news(text)

    print("Prediction:", label)
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Fake prob:   {fake_prob*100:.2f}%")
    print(f"Real prob:   {real_prob*100:.2f}%")
