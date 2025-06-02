""" Module containing model training logic """
import os
import pickle  # nosec
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    """ Main function performing model training"""
    vectorizer = TfidfVectorizer(max_features=1000)
    data_train = pd.read_csv('data/preprocessed/train.csv')
    data_train.dropna(inplace=True)
    print(data_train['cleaned'])
    x_train = vectorizer.fit_transform(data_train['cleaned'])
    y_train = data_train['Liked'].values

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    os.makedirs('models', exist_ok=True)  # <== Add this
    with open('models/c1_BoW.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    joblib.dump(model, 'models/c2_model.pkl')


if __name__ == '__main__':
    main()
