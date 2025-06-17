
import pandas as pd
import numpy as np
import pickle
from lib_ml.preprocess import clean_review
from sklearn.metrics import accuracy_score
import joblib
import pytest

RAW_NEG_WORDS = [
    "not", "no", "never", "don't", "can't", "won't", "couldn't", "didn't"
]
CLEANED_NEG_WORDS = [clean_review(w) for w in RAW_NEG_WORDS]

MIN_NEG_ACC = 0.70

@pytest.mark.ml_test("ML-6")   # ML-6: Model quality is sufficient on all important Data slices
def test_negation_slice_accuracy():

    df = pd.read_csv('data/preprocessed/test.csv')

    def contains_negation(text):
        return any(word in text.split() for word in CLEANED_NEG_WORDS)

    neg_df = df[df['cleaned'].apply(contains_negation)]
    assert len(neg_df) > 0, "No negation examples found in the cleaned test set!"

    with open('models/c1_BoW.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    model = joblib.load('models/c2_model.pkl')

    X_neg = vectorizer.transform(neg_df['cleaned'])
    y_neg = neg_df['Liked'].values
    preds = model.predict(X_neg)

    acc = accuracy_score(y_neg, preds)
    assert acc >= MIN_NEG_ACC, (
        f"Negation-slice accuracy too low: {acc:.2%} < {MIN_NEG_ACC:.2%}"
    )



@pytest.mark.ml_test("ML-5")   # ML-5: A simpler model is not better:
def test_simpler_model_not_better():
    df_train = pd.read_csv('data/preprocessed/train.csv')
    df_test = pd.read_csv('data/preprocessed/test.csv')

    majority_class = df_train['Liked'].mode()[0]

    y_true = df_test['Liked'].values
    y_baseline = np.full_like(y_true, fill_value=majority_class)
    acc_baseline = accuracy_score(y_true, y_baseline)

    with open('models/c1_BoW.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    model = joblib.load('models/c2_model.pkl')

    X_test = vectorizer.transform(df_test['cleaned'])
    y_pred = model.predict(X_test)
    acc_model = accuracy_score(y_true, y_pred)

    margin = 0.05
    assert acc_model >= acc_baseline + margin, (
        f"Model accuracy ({acc_model:.2%}) not at least {margin:.0%} better than baseline ({acc_baseline:.2%})"
    )

