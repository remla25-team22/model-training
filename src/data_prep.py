""" Module containing data preparation logic """
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from lib_ml.preprocess import clean_review


def main():
    """ Main data preperation function, prcesses and saves data for training """
    dataset = pd.read_csv(
            filepath_or_buffer='data/raw/a1_RestaurantReviews_HistoricDump.tsv',
            delimiter='\t',
            quoting=3
        )
    dataset['cleaned'] = dataset['Review'].apply(clean_review)
    dataset = dataset[['cleaned', 'Liked']]

    # First split into train and temp (test + val)
    train_set, temp_set = train_test_split(
            dataset,
            test_size=0.2,
            random_state=42,
            stratify=dataset['Liked']
        )

    # Then split temp into val and test (50/50 of remaining 20%)
    val_set, test_set = train_test_split(
            temp_set,
            test_size=0.5,
            random_state=42,
            stratify=temp_set['Liked']
        )
    os.makedirs('data/preprocessed', exist_ok=True)  # <== ADD THIS LINE

    train_set.to_csv('data/preprocessed/train.csv', index=False)
    val_set.to_csv('data/preprocessed/val.csv', index=False)
    test_set.to_csv('data/preprocessed/test.csv', index=False)


if __name__ == '__main__':
    main()
