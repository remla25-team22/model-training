""" Module containing model evaluation logic """
import json
import pickle  # nosec
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

from . import config


def main():
    """ Main funtion performing model evaluation """
    with open(f'{config.MODEL_DIR}/c1_BoW.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    model = joblib.load(f'{config.MODEL_DIR}/c2_model.pkl')

    data_test = pd.read_csv(f'{config.DATA_DIR}/preprocessed/test.csv', keep_default_na=False)
    x_test = vectorizer.transform(data_test['cleaned'])
    y_test = data_test['Liked'].values

    predictions = model.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    metrics = {
        'accuracy': acc,
        'confusion_matrix': cm.tolist()
    }
    print("This shows old value ", config.REPORT_DIR)
    with open(f'{config.REPORT_DIR}/metrics.json', 'w', encoding='UTF-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
