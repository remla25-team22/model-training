import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.discard('not')

def clean_review(review):
    if not isinstance(review, str):
        return ""
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    return ' '.join(ps.stem(w) for w in review if w not in stop_words)

def main():
    dataset = pd.read_csv('data/a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)
    dataset['cleaned'] = dataset['Review'].apply(clean_review)
    dataset[['cleaned', 'Liked']].to_csv('data/processed.csv', index=False)

if __name__ == '__main__':
    main()
