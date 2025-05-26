import pickle
import joblib
from src.data_prep import clean_review
import numpy as np

def predict(review: str) -> str:
    with open("models/c1_BoW.pkl", "rb") as f:
        cv = pickle.load(f)

    model = joblib.load("models/c2_model.pkl")

    vector = cv.transform([clean_review(review)]).toarray()
    prediction = model.predict(vector)[0]

    return "positive" if prediction == 1 else "negative"


if __name__ == "__main__":
    review = input("Give me a review:\n> ")
    sentiment = predict(review)
    print(f"The model believes the review is {sentiment}.")
