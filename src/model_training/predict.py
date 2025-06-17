""" Module containing prediction logic """
import pickle  # nosec
import joblib
from lib_ml.preprocess import clean_review

def predict(review_input: str) -> str:
    """ Function performing sentiment analysis on the given input string """
    with open("models/c1_BoW.pkl", "rb") as f:
        cv = pickle.load(f)  # nosec - file is known

    model = joblib.load("models/c2_model.pkl")

    vector = cv.transform([clean_review(review_input)]).toarray()
    prediction = model.predict(vector)[0]

    return "positive" if prediction == 1 else "negative"


if __name__ == "__main__":
    review = input("Give me a review:\n> ")
    SENTIMENT = predict(review)
    print(f"The model believes the review is {SENTIMENT}.")
