import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import json

def main():
    df = pd.read_csv('data/predictions.csv')
    acc = accuracy_score(df['true'], df['pred'])
    cm = confusion_matrix(df['true'], df['pred'])

    metrics = {'accuracy': acc, 'confusion_matrix': cm.tolist()}
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()
