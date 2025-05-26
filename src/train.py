import pandas as pd
import pickle
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def main():
    data = pd.read_csv('data/processed.csv')
    data.dropna(subset=['cleaned'], inplace=True)
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(data['cleaned']).toarray()
    y = data['Liked'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = GaussianNB()
    model.fit(X_train, y_train)

    with open('models/c1_BoW.pkl', 'wb') as f:
        pickle.dump(cv, f)
    joblib.dump(model, 'models/c2_model.pkl')

    pd.DataFrame({'true': y_test, 'pred': model.predict(X_test)}).to_csv('data/predictions.csv', index=False)

if __name__ == '__main__':
    main()
