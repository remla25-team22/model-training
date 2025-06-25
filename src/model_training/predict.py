""" Module containing prediction logic """
import pickle  # nosec
import joblib
from lib_ml.preprocess import clean_review
from . import config


def predict(review_input: str) -> str:
    """ Function performing sentiment analysis on the given input string """
    with open(f"{config.MODEL_DIR}/c1_BoW.pkl", "rb") as f:
        cv = pickle.load(f)

    model = joblib.load(f"{config.MODEL_DIR}/c2_model.pkl")

    vector = cv.transform([clean_review(review_input)]).toarray()
    prediction = model.predict(vector)[0]

    return "positive" if prediction == 1 else "negative"


if __name__ == "__main__":
    review = input("Give me a review:\n> ")
    SENTIMENT = predict(review)
    print(f"The model believes the review is {SENTIMENT}.")
