import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from lib_ml.preprocess import clean_review


def train_model():
    dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)

    nltk.download('stopwords')

    ps = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    corpus=[]

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    cv = CountVectorizer(max_features = 1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    bow_path = 'c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'c2_Classifier_Sentiment_Model') 
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy_score(y_test, y_pred)

def predict_sentiment():
    dataset = pd.read_csv('a2_RestaurantReviews_FreshDump.tsv', delimiter = '\t', quoting = 3)
    dataset.tail()

    # ps = PorterStemmer()

    # all_stopwords = stopwords.words('english')
    # all_stopwords.remove('not')
    # def clean_review(review):
    #     review = re.sub('[^a-zA-Z]', ' ', review)
    #     review = review.lower()
    #     review = review.split()
    #     review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    #     review = ' '.join(review)
    #     return review

    corpus=[]

    for i in range(0, 100):
        corpus.append(clean_review(dataset['Review'][i]))

    cvFile='c1_BoW_Sentiment_Model.pkl'

    cv = pickle.load(open(cvFile, "rb"))

    X_fresh = cv.transform(corpus).toarray()
    X_fresh.shape

    classifier = joblib.load('c2_Classifier_Sentiment_Model')

    y_pred = classifier.predict(X_fresh)
    print(y_pred)

    dataset['predicted_label'] = y_pred.tolist()
    dataset[dataset['predicted_label']==1]


    dataset.to_csv("c3_Predicted_Sentiments_Fresh_Dump.tsv", sep='\t', encoding='UTF-8', index=False)

    review = input("Give me an input to perform a sentiment analysis.\n>")

    processed_input = cv.transform([review]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    prediction_map = {
        0: "negative",
        1: "positive"
    }
    print(f"The model believes the review is {prediction_map[prediction]}.")

def main():
    train_model()
    predict_sentiment()

if __name__ == "__main__":
    main()