# Model Training

This repository contains the training pipeline for the Restaurant Sentiment Analysis project. It handles data preparation, model training, evaluation, and versioned release of trained models.

##  Features

- Loads and preprocesses restaurant review data
- Trains vectorizer and classifier components
- Stores model artifacts for use in the `model-service`
- Uses DVC to track data and models
- Includes CI/CD workflows to automate releases

##  Project Structure

- `src/` — Contains the core logic for data preparation, training, evaluation, and prediction
- `data/` — Contains raw data and tracked model artifacts
- `dvc.yaml` — Defines pipeline stages for DVC
- `requirements.txt` — Python dependencies
- `.github/workflows/` — GitHub Actions for release automation


##  Artifacts & Releases

Trained models (vectorizer + classifier) are stored and released via GitHub Releases. They are downloaded by the `model-service` at runtime based on the `MODEL_TAG`.

## Google Drive access
To ensure a successfull `dvc pull` , the command `cd ./.dvc && ./gdrive-access.sh` should be run.

##  Related Services

- [model-service](https://github.com/remla25-team22/model-service)
- [lib-ml](https://github.com/remla25-team22/lib-ml)
- [operation](https://github.com/remla25-team22/operation)

