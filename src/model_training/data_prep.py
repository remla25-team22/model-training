""" Module containing data prep logic """
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from lib_ml.preprocess import clean_review

from . import config
def main():
    """ Main data preperation function, prcesses and saves data for training """
    dataset = pd.read_csv(
        filepath_or_buffer=f'{config.DATA_DIR}/raw/a1_RestaurantReviews_HistoricDump.tsv',
        delimiter='\t',
        quoting=3,
        keep_default_na=False
    )

    # Clean reviews
    dataset['cleaned'] = dataset['Review'].apply(clean_review)

    # Remove rows with NaN values (if cleaning caused any)
    dataset = dataset[['cleaned', 'Liked']].dropna()

    # First split into train and temp (test + val)
    train_set, temp_set = train_test_split(
        dataset,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=dataset['Liked']
    )

    # Then split temp into val and test (50/50 of remaining 20%)
    val_set, test_set = train_test_split(
        temp_set,
        test_size=0.5,
        random_state=config.RANDOM_SEED,
        stratify=temp_set['Liked']
    )

    os.makedirs(f'{config.DATA_DIR}/preprocessed', exist_ok=True)
    train_set.to_csv(f'{config.DATA_DIR}/preprocessed/train.csv', index=False)
    val_set.to_csv(f'{config.DATA_DIR}/preprocessed/val.csv', index=False)
    test_set.to_csv(f'{config.DATA_DIR}/preprocessed/test.csv', index=False)


if __name__ == '__main__':
    main()
