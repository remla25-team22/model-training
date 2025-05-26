
import pytest
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib
from lib_ml.preprocess import clean_review

def test_input_data_columns():
    df = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)
    assert 'Review' in df.columns
    assert df.shape[0] > 0

def test_no_empty_reviews():
    df = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)
    assert all(df['Review'].str.strip() != '')

def test_model_output_shape():
    sample_reviews = ["The food was great", "I did not like the service"]
    cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', "rb"))
    model = joblib.load('c2_Classifier_Sentiment_Model')
    X = cv.transform([clean_review(r) for r in sample_reviews]).toarray()
    y_pred = model.predict(X)
    assert y_pred.shape[0] == 2

def test_model_files_exist():
    assert os.path.exists('c1_BoW_Sentiment_Model.pkl')
    assert os.path.exists('c2_Classifier_Sentiment_Model')

def test_prediction_range():
    sample_reviews = ["Excellent food", "Awful experience"]
    cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', "rb"))
    model = joblib.load('c2_Classifier_Sentiment_Model')
    X = cv.transform([clean_review(r) for r in sample_reviews]).toarray()
    y_pred = model.predict(X)
    assert set(y_pred).issubset({0, 1})

# Mutamorphic testing 
@pytest.mark.parametrize("original,synonym", [
    ("The food was amazing", "The food was good"),
    ("I didn't like it", "I did not like it"),
    ("Service was bad", "Service was awful")
])
def test_mutamorphic_consistency(original, synonym):
    cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', "rb"))
    model = joblib.load('c2_Classifier_Sentiment_Model')
    original_vec = cv.transform([clean_review(original)]).toarray()
    synonym_vec = cv.transform([clean_review(synonym)]).toarray()
    pred_original = model.predict(original_vec)[0]
    pred_synonym = model.predict(synonym_vec)[0]
    assert pred_original == pred_synonym

def test_model_inference_time():
    import time
    sample_review = "This was a wonderful place with amazing staff"
    cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', "rb"))
    model = joblib.load('c2_Classifier_Sentiment_Model')
    X = cv.transform([clean_review(sample_review)]).toarray()
    start = time.time()
    _ = model.predict(X)
    end = time.time()
    assert (end - start) < 0.5 
