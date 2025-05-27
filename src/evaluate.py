import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib

def main():

    with open('models/c1_BoW.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    model = joblib.load('models/c2_model.pkl')

    data_test = pd.read_csv('data/preprocessed/test.csv')
    X_test = vectorizer.transform(data_test['cleaned'])
    y_test = data_test['Liked'].values

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    metrics = {
        'accuracy': acc,
        'confusion_matrix': cm.tolist()
    }

    with open('data/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()
